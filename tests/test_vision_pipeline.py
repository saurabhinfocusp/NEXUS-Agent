"""Integration tests: real CellPose segmentation + VGG16 embedding
(Constitution Art. XII §2). Marked integration since first run downloads
model weights (CellPose ~1.15GB, VGG16 ~528MB) if not already cached.
"""

import numpy as np
import pytest
from skimage.draw import disk

from nexus_agent.vision.embedding import EMBEDDING_DIM, Vgg16Embedder
from nexus_agent.vision.segmentation import cells_from_mask, crop_patch, segment_cells, stardist_segment

pytestmark = pytest.mark.integration


def _synthetic_blob_image(n_blobs: int = 5, size: int = 128, seed: int = 0) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(n_blobs):
        cy, cx = rng.integers(20, size - 20, size=2)
        rr, cc = disk((cy, cx), 10, shape=image.shape)
        image[rr, cc] = 200
    return image


def test_segment_cells_finds_synthetic_blobs():
    image = _synthetic_blob_image(n_blobs=5)
    mask = segment_cells(image)

    assert mask.shape == image.shape
    assert mask.max() > 0  # found at least one cell


def test_cells_from_mask_produces_contract_ready_geometry():
    image = _synthetic_blob_image(n_blobs=4)
    mask = segment_cells(image)
    cells = cells_from_mask(mask)

    assert len(cells) == mask.max()
    for cell in cells:
        assert cell["cell_id"]
        x, y = cell["centroid_xy"]
        assert 0 <= x < image.shape[1]
        assert 0 <= y < image.shape[0]
        assert len(cell["mask_polygon"]) > 0

        patch = crop_patch(image, cell["bbox"])
        assert patch.ndim == 3 and patch.shape[2] == 3
        assert patch.dtype == np.uint8


def test_stardist_fallback_is_explicitly_deferred():
    with pytest.raises(NotImplementedError):
        stardist_segment(_synthetic_blob_image())


def test_vgg16_embedder_produces_zero_padded_1024dim_vector():
    patch = (np.random.default_rng(0).random((40, 35, 3)) * 255).astype(np.uint8)
    embedder = Vgg16Embedder()

    vector = embedder.embed_patch(patch)

    assert vector.shape == (EMBEDDING_DIM,)
    assert np.count_nonzero(vector[:512]) > 0  # real conv5_3 activations
    assert np.all(vector[512:] == 0.0)  # zero-padded to the fixed 1024-dim contract
    assert embedder.model_version == "vgg16-conv5_3-zeropad1024-v1"
