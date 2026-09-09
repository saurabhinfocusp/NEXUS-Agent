"""Parse uploaded files into the in-memory shapes the pipeline expects.

`agents/vision.py`'s real path needs a plain `np.ndarray` (via
`data/object_store.py::put_array`/`get_array`, which round-trips through
`.npy`, not the original file format); `agents/analyst.py`'s real path
needs an `AnnData` (via `put_anndata`/`get_anndata`, `.h5ad`). A user
uploads a PNG/JPEG/TIFF image and an `.h5ad` expression file -- this module
is the one place those raw bytes get turned into what the rest of the
pipeline already knows how to store and consume.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import skimage.io


def read_image_upload(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG/TIFF (or anything `skimage.io.imread` supports) bytes
    into an array `vision/segmentation.py::segment_cells`/`crop_patch` can
    consume -- one code path instead of branching on file extension.
    """
    return skimage.io.imread(io.BytesIO(data))


def read_expression_upload(data: bytes) -> ad.AnnData:
    """Decode `.h5ad` bytes into an `AnnData`. `anndata`'s HDF5 reader needs
    a real file path, not a file-like object, so this round-trips through a
    temp file -- same pattern `data/object_store.py::put_anndata`/
    `get_anndata` already use.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "upload.h5ad"
        tmp_path.write_bytes(data)
        return ad.read_h5ad(tmp_path)
