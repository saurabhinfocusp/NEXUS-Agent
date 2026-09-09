"""Fast, no-infra tests for webapp/uploads.py's file-parsing helpers."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from anndata import AnnData
from PIL import Image

from nexus_agent.webapp.uploads import read_expression_upload, read_image_upload


def test_read_image_upload_decodes_png():
    array = (np.random.default_rng(0).integers(0, 255, size=(32, 32), dtype=np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")

    decoded = read_image_upload(buffer.getvalue())

    assert decoded.shape == (32, 32)
    assert np.array_equal(decoded, array)


def test_read_image_upload_decodes_tiff():
    array = np.random.default_rng(1).integers(0, 255, size=(16, 16), dtype=np.uint8)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array)

    decoded = read_image_upload(buffer.getvalue())

    assert decoded.shape == (16, 16)
    assert np.array_equal(decoded, array)


def test_read_expression_upload_decodes_h5ad():
    rng = np.random.default_rng(2)
    adata = AnnData(
        X=rng.poisson(3, size=(4, 6)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test.h5ad"
        adata.write_h5ad(tmp_path)
        data = tmp_path.read_bytes()

    decoded = read_expression_upload(data)

    assert decoded.n_obs == 4
    assert decoded.n_vars == 6
