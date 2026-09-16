"""Parse uploaded files into the in-memory shapes the pipeline expects.

`agents/vision.py`'s real path needs a plain `np.ndarray` (via
`data/object_store.py::put_array`/`get_array`, which round-trips through
`.npy`, not the original file format); `agents/analyst.py`'s real path
needs an `AnnData` (via `put_anndata`/`get_anndata`, `.h5ad`). A user
uploads a PNG/JPEG/TIFF image and one or more expression files -- this
module is the one place those raw bytes get turned into what the rest of
the pipeline already knows how to store and consume.

Expression files aren't always native AnnData, and aren't always a single
file: `read_expression_upload` accepts either one file -- `.h5ad`,
`.loom`, a 10x Genomics `.h5` feature-barcode matrix, or a zipped/tarred
10x Genomics mtx bundle -- or several loose files selected together in
the browser (`matrix.mtx`/`barcodes.tsv`/`features.tsv`, each optionally
`.gz`, with no archive step required). Either way it converts to the same
`AnnData` shape, so nothing downstream of this module has to know which
format or file count a given upload started as.

Seurat/SingleCellExperiment `.rds` objects are deliberately NOT read
here -- R's native serialization format needs a real R+Bioconductor
runtime to parse correctly (rpy2, zellkonverter), which this project
doesn't otherwise depend on, and that dependency would be heavy and
fragile to keep working across R/Seurat versions for a single upload
path. Convert on the R side first instead, e.g.
`SeuratDisk::SaveH5Seurat()` + `Convert(dest="h5ad")`, or
`sceasy::convertFormat(obj, from="seurat", to="anndata")`, then upload
the resulting `.h5ad`.

A 10x Genomics Loupe `.cloupe` file can't be read at all, by anything
other than 10x's own Loupe Browser -- unlike `.rds`, its format is
proprietary and undocumented with no reader in any language to build on,
official or otherwise. A `.cloupe` is generated *from* CellRanger's
feature-barcode matrix, so `_read_single_expression_file` raises a
pointed error steering the user back to that `*_feature_bc_matrix.h5` or
mtx bundle instead, both already supported.

Either upload can also be a single plain-gzipped file (`image.tiff.gz`,
`expression.h5ad.gz`, ...) -- `_maybe_gunzip` transparently decompresses
before dispatching on whatever extension is underneath. This is separate
from the `.tar.gz`/`.tgz` *archive* case (multiple files bundled, handled
by `_extract_archive`) and from a loose `matrix.mtx.gz`/`barcodes.tsv.gz`/
`features.tsv.gz` selection (already `.gz`-aware via `scanpy.read_10x_mtx`
itself, unrelated to this module's own gunzip step).
"""

from __future__ import annotations

import gzip
import io
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import scanpy as sc
import skimage.io


def _maybe_gunzip(filename: str, data: bytes) -> tuple[str, bytes]:
    """If `filename` ends in a plain `.gz` -- not `.tar.gz`/`.tgz`, which
    are handled as archives elsewhere -- decompress `data` and return the
    filename with that suffix stripped, so the caller can dispatch on
    whatever real format is underneath (`.png.gz` -> `.png`, `.h5ad.gz` ->
    `.h5ad`, ...).
    """
    lowered = filename.lower()
    if lowered.endswith(".gz") and not lowered.endswith(".tar.gz"):
        return filename[: -len(".gz")], gzip.decompress(data)
    return filename, data


def read_image_upload(data: bytes, filename: str) -> np.ndarray:
    """Decode PNG/JPEG/TIFF (or anything `skimage.io.imread` supports) bytes
    into an array `vision/segmentation.py::segment_cells`/`crop_patch` can
    consume -- one code path instead of branching on file extension.

    Reads from a real temp file with `filename`'s suffix rather than an
    in-memory `BytesIO`: `imageio` (skimage's backend) picks its decoder
    plugin from the file extension, and without one it falls back to
    sniffing the format from the bytes alone, which isn't reliable for
    every real-world image (observed failing outright -- "Could not find a
    backend" -- for at least some TIFFs that decode fine from a path).
    """
    filename, data = _maybe_gunzip(filename, data)
    suffix = Path(filename).suffix or ".png"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"upload{suffix}"
        tmp_path.write_bytes(data)
        return skimage.io.imread(tmp_path)


_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


def read_expression_upload(files: list[tuple[str, bytes]]) -> ad.AnnData:
    """Decode one or more uploaded expression files into an `AnnData`.

    A single file is dispatched on its extension (`.h5ad`, `.loom`, a 10x
    `.h5` matrix, or a `.zip`/`.tar.gz` mtx bundle). Multiple files are
    assumed to be a 10x mtx bundle's components selected loose in the
    browser (no archive step) -- each is written into one directory under
    its own basename and handed to `scanpy.read_10x_mtx` exactly like the
    single-archive path, just skipping extraction.
    """
    if not files:
        raise ValueError("no expression file provided")
    if len(files) == 1:
        filename, data = files[0]
        return _read_single_expression_file(filename, data)
    return _read_10x_mtx_components(files)


def _read_single_expression_file(filename: str, data: bytes) -> ad.AnnData:
    name = filename.lower()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        if name.endswith(".h5ad"):
            tmp_path = tmp_dir_path / "upload.h5ad"
            tmp_path.write_bytes(data)
            return ad.read_h5ad(tmp_path)

        if name.endswith(".loom"):
            tmp_path = tmp_dir_path / "upload.loom"
            tmp_path.write_bytes(data)
            return ad.read_loom(tmp_path)

        if name.endswith(".h5"):
            # A 10x Genomics feature-barcode matrix (CellRanger's
            # `*_feature_bc_matrix.h5`) -- distinct from `.h5ad`, which is
            # matched first above.
            tmp_path = tmp_dir_path / "upload.h5"
            tmp_path.write_bytes(data)
            return sc.read_10x_h5(tmp_path)

        if name.endswith(_ARCHIVE_SUFFIXES):
            extract_dir = tmp_dir_path / "extracted"
            extract_dir.mkdir()
            _extract_archive(data, name, extract_dir)
            return _read_extracted_expression(extract_dir)

        if name.endswith(".gz"):
            inner_name, inner_data = _maybe_gunzip(filename, data)
            return _read_single_expression_file(inner_name, inner_data)

        if name.endswith(".cloupe"):
            raise ValueError(
                f"{filename!r} is a 10x Genomics Loupe (.cloupe) file -- Loupe's format is "
                "proprietary and undocumented, with no reader (official or third-party, in any "
                "language) outside 10x's own Loupe Browser, so it can't be parsed here. A .cloupe "
                "is generated *from* CellRanger's feature-barcode matrix -- upload that instead: "
                "the `*_feature_bc_matrix.h5` (10x .h5, already supported) or the matching "
                "`filtered_feature_bc_matrix/` mtx bundle (matrix.mtx + barcodes.tsv + "
                "features.tsv, also already supported) that CellRanger wrote alongside it."
            )

        raise ValueError(
            f"unsupported expression file type: {filename!r} -- expected .h5ad, "
            ".loom, a 10x Genomics .h5 feature-barcode matrix, a .zip/.tar.gz of a "
            "10x Genomics mtx bundle, matrix.mtx + barcodes.tsv + features.tsv "
            "selected together, or any of those gzipped"
        )


def _read_10x_mtx_components(files: list[tuple[str, bytes]]) -> ad.AnnData:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mtx_dir = Path(tmp_dir)
        for filename, data in files:
            # `Path(...).name` guards against a path-y filename (e.g. a
            # browser-supplied relative path) escaping `mtx_dir`.
            (mtx_dir / Path(filename).name).write_bytes(data)
        return _read_extracted_expression(mtx_dir)


def _extract_archive(data: bytes, name: str, dest: Path) -> None:
    """Extract a zip/tar archive into `dest`, rejecting any member path that
    would escape it ("zip slip") -- the archive's contents are
    user-uploaded and otherwise untrusted.
    """
    buffer = io.BytesIO(data)
    dest_resolved = dest.resolve()

    if name.endswith(".zip"):
        with zipfile.ZipFile(buffer) as archive:
            for member in archive.namelist():
                if not (dest / member).resolve().is_relative_to(dest_resolved):
                    raise ValueError(f"unsafe path in archive: {member!r}")
            archive.extractall(dest)
        return

    with tarfile.open(fileobj=buffer) as archive:
        # Python 3.12's "data" extraction filter (PEP 706) rejects unsafe
        # paths, device files, and similar for us.
        archive.extractall(dest, filter="data")


_SINGLE_FILE_READERS: tuple[tuple[str, Callable[[Path], ad.AnnData]], ...] = (
    (".h5ad", ad.read_h5ad),
    # A lambda, not `ad.read_loom` directly -- referencing that attribute at
    # all (even without calling it) raises anndata's loom deprecation
    # warning, which would otherwise fire on every import of this module
    # instead of only when a .loom is actually read.
    (".loom", lambda path: ad.read_loom(path)),
    (".h5", sc.read_10x_h5),
)


def _read_extracted_expression(root: Path) -> ad.AnnData:
    """Figure out what's actually in `root` -- the contents of an extracted
    archive, or loose files assembled directly into it -- and read it.

    A 10x mtx bundle's files may sit directly in `root` (the loose-file
    upload path) or inside a wrapping folder (an archive extracted flat vs.
    with its own top-level directory), so this walks the tree for whichever
    directory actually holds `matrix.mtx[.gz]`, since `read_10x_mtx` needs
    that exact directory, not an ancestor of it. Failing that, an archive
    isn't necessarily an mtx bundle at all -- a `.zip` can just as well wrap
    a single `.h5ad`/`.loom`/10x `.h5` file (e.g. `expression_matrix.zip`
    containing one `.h5ad`) -- so this falls back to whichever of those it
    can find, in that preference order.
    """
    for candidate in sorted(root.rglob("matrix.mtx*")):
        adata = sc.read_10x_mtx(candidate.parent)
        adata.var_names_make_unique()
        return adata

    for suffix, reader in _SINGLE_FILE_READERS:
        matches = sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())
        if matches:
            return reader(matches[0])

    raise ValueError(
        "no recognizable expression file found among the uploaded file(s) -- expected "
        "a 10x Genomics mtx bundle (matrix.mtx + barcodes.tsv + features.tsv), or a "
        "single .h5ad, .loom, or 10x Genomics .h5 feature-barcode matrix"
    )
