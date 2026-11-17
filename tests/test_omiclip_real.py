"""OmiCLIP against its real checkpoint (Constitution Art. XII §2 as amended
in v1.3). Integration: the first run downloads 7.7 GB from Hugging Face,
and loading takes ~2.5 GB of memory.

These check that the pinned weights load, produce the contract shapes and
behave like a trained CLIP model -- they don't validate OmiCLIP on any
tissue, which needs labelled data.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.integration

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "examples" / "sample_image.png"


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("open_clip")
    from nexus_agent.vision.omiclip import OmiclipEmbedder

    return OmiclipEmbedder(device="cpu")


@pytest.fixture(scope="module")
def tiles():
    image = np.asarray(Image.open(SAMPLE_IMAGE).convert("RGB"))
    h, w = image.shape[:2]
    size = min(h, w) // 2
    return [image[r : r + size, c : c + size] for r in (0, h - size) for c in (0, w - size)]


def test_embeddings_meet_the_vision_contract(embedder, tiles):
    from nexus_agent.vision.embedding import EMBEDDING_DIM
    from nexus_agent.vision.omiclip import OMICLIP_DIM

    embeddings = embedder.embed_patches(tiles)

    assert embeddings.shape == (len(tiles), EMBEDDING_DIM)
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings[:, :OMICLIP_DIM], axis=1), 1.0, atol=1e-5)
    assert (embeddings[:, OMICLIP_DIM:] == 0).all()
    np.testing.assert_allclose(embedder.embed_patch(tiles[0]), embeddings[0], atol=1e-5)  # batching-invariant


def test_text_latents_are_normalized_and_distinguish_gene_programs(embedder):
    immune = "PTPRC CD3E CD3D CD2 CD8A CD4 MS4A1 CD79A CD19 LYZ CD68 CD14"
    epithelial = "EPCAM KRT8 KRT18 KRT19 CDH1 CLDN4 KRT7 MUC1 ELF3 SPINT2 CLDN7 KRT5"

    latents = embedder.text_latents([immune, immune, epithelial])

    assert latents.shape == (3, 768)
    np.testing.assert_allclose(np.linalg.norm(latents, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(latents[0], latents[1], atol=1e-5)
    assert latents[0] @ latents[2] < 0.999  # different programs, different latents


def test_vit_gradcam_runs_on_the_real_encoder(embedder, tiles):
    from nexus_agent.xai.gradcam import grad_cam_plusplus_vit

    target = embedder.text_latents(["EPCAM KRT8 KRT18 KRT19 CDH1"])[0]
    heatmap = grad_cam_plusplus_vit(embedder, tiles[0], target)

    assert heatmap.shape == tiles[0].shape[:2]
    assert np.isfinite(heatmap).all() and heatmap.max() <= 1.0 and heatmap.min() >= 0.0
    assert heatmap.std() > 0  # it localizes something
