"""Fast, no-infra tests for webapp/uploads.py's file-parsing helpers."""

import gzip
import io
import tarfile
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import tifffile
from anndata import AnnData
from PIL import Image

from nexus_agent.webapp.uploads import read_expression_upload, read_image_upload


def test_read_image_upload_decodes_png():
    array = (np.random.default_rng(0).integers(0, 255, size=(32, 32), dtype=np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")

    decoded = read_image_upload(buffer.getvalue(), "image.png")

    assert decoded.shape == (32, 32)
    assert np.array_equal(decoded, array)


def test_read_image_upload_decodes_tiff():
    array = np.random.default_rng(1).integers(0, 255, size=(16, 16), dtype=np.uint8)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array)

    decoded = read_image_upload(buffer.getvalue(), "image.tiff")

    assert decoded.shape == (16, 16)
    assert np.array_equal(decoded, array)


def test_read_image_upload_decodes_jpeg():
    array = np.random.default_rng(5).integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")

    decoded = read_image_upload(buffer.getvalue(), "image.jpg")

    assert decoded.shape == (24, 24, 3)


def test_read_image_upload_decodes_gzipped_png():
    array = np.random.default_rng(6).integers(0, 255, size=(20, 20), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    gzipped = gzip.compress(buffer.getvalue())

    decoded = read_image_upload(gzipped, "image.png.gz")

    assert decoded.shape == (20, 20)
    assert np.array_equal(decoded, array)


def _synthetic_adata(n_obs: int = 4, n_vars: int = 6, seed: int = 2) -> AnnData:
    rng = np.random.default_rng(seed)
    return AnnData(
        X=rng.poisson(3, size=(n_obs, n_vars)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
    )


def test_read_expression_upload_decodes_h5ad():
    adata = _synthetic_adata()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.h5ad"
        adata.write_h5ad(tmp_path)
        data = tmp_path.read_bytes()

    decoded = read_expression_upload([("test.h5ad", data)])

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def test_read_expression_upload_decodes_gzipped_h5ad():
    adata = _synthetic_adata()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.h5ad"
        adata.write_h5ad(tmp_path)
        data = gzip.compress(tmp_path.read_bytes())

    decoded = read_expression_upload([("test.h5ad.gz", data)])

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def test_read_expression_upload_decodes_loom():
    import loompy

    n_obs, n_vars = 4, 6
    adata = _synthetic_adata(n_obs=n_obs, n_vars=n_vars)
    # loompy's own convention (rows=genes, columns=cells), written directly
    # with loompy rather than `AnnData.write_loom` -- matches what a real
    # loom producer (e.g. R's loomR, or loompy itself) hands us, and
    # sidesteps an unrelated anndata/loompy version incompatibility in
    # `AnnData.write_loom` for an unnamed default layer.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.loom"
        loompy.create(
            str(tmp_path),
            adata.X.T,
            row_attrs={"Gene": np.array(adata.var_names)},
            col_attrs={"CellID": np.array(adata.obs_names)},
        )
        data = tmp_path.read_bytes()

    decoded = read_expression_upload([("test.loom", data)])

    assert decoded.n_obs == n_obs
    assert decoded.n_vars == n_vars


def _write_10x_v3_h5(path: Path, n_cells: int = 4, n_genes: int = 6, seed: int = 3) -> None:
    """Build a minimal CellRanger v3+ feature-barcode matrix.h5, matching
    the schema `scanpy.read_10x_h5`'s `_read_v3_10x_h5` expects -- a
    "matrix" group with a CSR-compatible data/indices/indptr/shape plus
    barcodes and a "features" subgroup with id/name/feature_type.
    """
    rng = np.random.default_rng(seed)
    dense = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    sparse = scipy.sparse.csr_matrix(dense)

    with h5py.File(path, "w") as f:
        matrix = f.create_group("matrix")
        matrix.create_dataset("data", data=sparse.data)
        matrix.create_dataset("indices", data=sparse.indices)
        matrix.create_dataset("indptr", data=sparse.indptr)
        matrix.create_dataset("shape", data=np.array([n_genes, n_cells]))
        matrix.create_dataset(
            "barcodes", data=np.array([f"cell_{i}".encode() for i in range(n_cells)])
        )
        features = matrix.create_group("features")
        features.create_dataset(
            "id", data=np.array([f"gene_{i}".encode() for i in range(n_genes)])
        )
        features.create_dataset(
            "name", data=np.array([f"gene_{i}".encode() for i in range(n_genes)])
        )
        features.create_dataset(
            "feature_type", data=np.array([b"Gene Expression"] * n_genes)
        )


def test_read_expression_upload_decodes_10x_h5():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "filtered_feature_bc_matrix.h5"
        _write_10x_v3_h5(tmp_path)
        data = tmp_path.read_bytes()

    decoded = read_expression_upload([("filtered_feature_bc_matrix.h5", data)])

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def _write_10x_mtx_dir(directory: Path, n_cells: int = 4, n_genes: int = 6, seed: int = 4) -> None:
    import gzip

    adata = _synthetic_adata(n_obs=n_cells, n_vars=n_genes, seed=seed)
    gene_ids = [f"ENSG{i}" for i in range(n_genes)]

    directory.mkdir(parents=True, exist_ok=True)
    matrix = scipy.sparse.csr_matrix(adata.X).T.tocoo()  # 10x mtx is genes x cells
    with gzip.open(directory / "matrix.mtx.gz", "wb") as fh:
        fh.write(b"%%MatrixMarket matrix coordinate integer general\n")
        fh.write(f"{matrix.shape[0]} {matrix.shape[1]} {matrix.nnz}\n".encode())
        for r, c, v in zip(matrix.row, matrix.col, matrix.data):
            fh.write(f"{r + 1} {c + 1} {int(v)}\n".encode())
    with gzip.open(directory / "barcodes.tsv.gz", "wt") as fh:
        for name in adata.obs_names:
            fh.write(f"{name}\n")
    with gzip.open(directory / "features.tsv.gz", "wt") as fh:
        for gene_id, name in zip(gene_ids, adata.var_names):
            fh.write(f"{gene_id}\t{name}\tGene Expression\n")


def test_read_expression_upload_decodes_10x_mtx_zip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mtx_dir = Path(tmp_dir) / "filtered_feature_bc_matrix"
        _write_10x_mtx_dir(mtx_dir)

        zip_path = Path(tmp_dir) / "sample.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for file in mtx_dir.iterdir():
                archive.write(file, arcname=f"filtered_feature_bc_matrix/{file.name}")
        data = zip_path.read_bytes()

    decoded = read_expression_upload([("sample.zip", data)])

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def test_read_expression_upload_decodes_zip_wrapping_single_h5ad():
    """A zip need not be a 10x mtx bundle -- e.g. `expression_matrix.zip`
    wrapping one `.h5ad` should also be read, falling back past the
    mtx-bundle check to a single recognizable file inside.
    """
    adata = _synthetic_adata()
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5ad_path = Path(tmp_dir) / "expression_matrix.h5ad"
        adata.write_h5ad(h5ad_path)

        zip_path = Path(tmp_dir) / "expression_matrix.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(h5ad_path, arcname="expression_matrix.h5ad")
        data = zip_path.read_bytes()

    decoded = read_expression_upload([("expression_matrix.zip", data)])

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def test_read_expression_upload_decodes_10x_mtx_tar_gz():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mtx_dir = Path(tmp_dir) / "filtered_feature_bc_matrix"
        _write_10x_mtx_dir(mtx_dir)

        tar_path = Path(tmp_dir) / "sample.tar.gz"
        with tarfile.open(tar_path, "w:gz") as archive:
            archive.add(mtx_dir, arcname="filtered_feature_bc_matrix")
        data = tar_path.read_bytes()

    decoded = read_expression_upload([("sample.tar.gz", data)])

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def test_read_expression_upload_decodes_loose_10x_mtx_components():
    """The frontend lets a user multi-select matrix.mtx/barcodes.tsv/
    features.tsv directly, with no archive step -- `read_expression_upload`
    should assemble those into the same result as the zipped/tarred path.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mtx_dir = Path(tmp_dir) / "components"
        _write_10x_mtx_dir(mtx_dir)
        files = [(f.name, f.read_bytes()) for f in sorted(mtx_dir.iterdir())]

    decoded = read_expression_upload(files)

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6


def test_read_expression_upload_rejects_no_files():
    with pytest.raises(ValueError, match="no expression file provided"):
        read_expression_upload([])


def test_read_expression_upload_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="unsupported expression file type"):
        read_expression_upload([("notes.txt", b"not a real file")])


def test_read_expression_upload_rejects_cloupe_with_pointed_message():
    with pytest.raises(ValueError, match="proprietary and undocumented"):
        read_expression_upload([("sample.cloupe", b"not a real file")])


def test_read_expression_upload_rejects_loose_files_missing_matrix():
    with pytest.raises(ValueError, match="no recognizable expression file found"):
        read_expression_upload([("barcodes.tsv", b"cell_0\n"), ("features.tsv", b"gene_0\tGene0\n")])


def test_read_expression_upload_rejects_zip_slip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../etc/passwd", "pwned")
    with pytest.raises(ValueError, match="unsafe path in archive"):
        read_expression_upload([("evil.zip", buffer.getvalue())])
