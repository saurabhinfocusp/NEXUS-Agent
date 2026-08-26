import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData

from nexus_agent.data.container import (
    REGISTRATION_ERROR_THRESHOLD_UM,
    build_spatialdata,
    check_registration,
)


def test_check_registration_passes_when_aligned():
    coords = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 5.0]])
    result = check_registration(coords, coords.copy())

    assert result.max_error_um == pytest.approx(0.0)
    assert result.registered is True


def test_check_registration_fails_when_offset_exceeds_threshold():
    mask_coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    expression_coords = mask_coords + np.array([REGISTRATION_ERROR_THRESHOLD_UM + 5.0, 0.0])

    result = check_registration(mask_coords, expression_coords)

    assert result.registered is False


def test_check_registration_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        check_registration(np.zeros((3, 2)), np.zeros((4, 2)))


def _synthetic_sample(n_cells: int = 4):
    """A tiny (3, 16, 16) image, a matching (16, 16) label mask with n_cells
    distinct quadrant labels, and an AnnData whose obs links each row back
    to a mask label via the spatialdata TableModel `region`/`instance_id`
    convention.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(3, 16, 16), dtype=np.uint8)

    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[0:8, 0:8] = 1
    mask[0:8, 8:16] = 2
    mask[8:16, 0:8] = 3
    mask[8:16, 8:16] = 4
    mask = mask[:n_cells + (4 - n_cells)]  # no-op guard, keeps 4 labels for n_cells<=4

    obs = pd.DataFrame(
        {
            "region": ["cells"] * n_cells,
            "instance_id": list(range(1, n_cells + 1)),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = AnnData(X=rng.random((n_cells, 5), dtype=np.float32), obs=obs)
    return image, mask, adata


def test_build_spatialdata_binds_image_mask_and_table():
    image, mask, adata = _synthetic_sample()

    sdata = build_spatialdata(image, mask, adata)

    assert isinstance(sdata, SpatialData)
    assert "image" in sdata.images
    assert "cells" in sdata.labels
    assert sdata.tables["table"].n_obs == adata.n_obs
