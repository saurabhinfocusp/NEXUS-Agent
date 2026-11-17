"""Tests for the OmiCLIP integration that need no weights: gene
sentences, the agreement score, the ViT Grad-CAM++ plumbing (against a tiny
randomly initialized open_clip ViT, not OmiCLIP), embedder selection, and
how Analyst and the report surface agreement. `tests/test_omiclip_real.py`
exercises the real checkpoint.
"""

import numpy as np
import pytest
import torch

from nexus_agent.analyst.agreement import retrieval_agreement
from nexus_agent.shared.config import settings
from nexus_agent.vision.embedding import EMBEDDING_DIM, Vgg16Embedder, configured_embedder_version
from nexus_agent.vision.omiclip import (
    OMICLIP_DIM,
    OmiclipEmbedder,
    gene_sentences,
    image_latent_from_embedding,
    is_omiclip_version,
)


def test_gene_sentence_lists_most_expressed_genes_first_and_skips_zeros():
    genes = ["CD3E", "MS4A1", "ACTB", "CD8A"]
    expression = np.array([[5.0, 0.0, 9.0, 1.0], [0.0, 0.0, 0.0, 2.0]])

    assert gene_sentences(genes, expression, n_genes=3) == ["ACTB CD3E CD8A", "CD8A"]


def test_gene_sentence_drops_names_loki_drops():
    genes = ["CD3E", "RP11-34P13.7", "HLA-A", "CD8A"]
    expression = np.array([[1.0, 9.0, 8.0, 2.0]])

    assert gene_sentences(genes, expression) == ["CD8A CD3E"]


def test_gene_sentence_is_invariant_to_per_cell_normalization():
    rng = np.random.default_rng(0)
    counts = rng.poisson(5, size=(3, 60)).astype(float) + rng.random((3, 60))
    genes = [f"G{i}" for i in range(60)]
    normalized = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    assert gene_sentences(genes, counts) == gene_sentences(genes, normalized)
    assert all(len(s.split()) == 50 for s in gene_sentences(genes, counts))


# --- agreement -----------------------------------------------------------------


def _unit(rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_agreement_is_one_when_every_image_picks_out_its_own_profile():
    latents = _unit(np.eye(4))

    np.testing.assert_allclose(retrieval_agreement(latents, latents), np.ones(4))


def test_agreement_is_zero_when_an_image_matches_every_other_profile_better():
    images = _unit([[1, 0], [0, 1], [0, 1]])
    texts = _unit([[-1, 0.1], [0.1, 1], [0, 1]])

    agreement = retrieval_agreement(images, texts)

    assert agreement[0] == 0.0  # cell 0's image is farthest from its own profile


def test_agreement_counts_ties_as_half_and_is_chance_for_uninformative_images():
    images = _unit(np.ones((5, 3)))  # every image identical: no information
    texts = _unit(np.random.default_rng(0).normal(size=(5, 3)))

    agreement = retrieval_agreement(images, _unit(np.ones((5, 3))))
    assert agreement == pytest.approx(np.full(5, 0.5))
    assert retrieval_agreement(images, texts).mean() == pytest.approx(0.5)


def test_agreement_is_undefined_for_a_single_cell_and_rejects_bad_shapes():
    assert np.isnan(retrieval_agreement(np.ones((1, 3)), np.ones((1, 3)))).all()
    with pytest.raises(ValueError):
        retrieval_agreement(np.ones((2, 3)), np.ones((3, 3)))


# --- embedder plumbing ---------------------------------------------------------


def test_omiclip_is_the_default_encoder_and_vgg16_is_selectable(monkeypatch):
    assert settings.vision_embedder == "omiclip"
    assert configured_embedder_version() == OmiclipEmbedder.model_version
    assert is_omiclip_version(OmiclipEmbedder.model_version)
    assert "8e08d3a" in OmiclipEmbedder.model_version  # the pinned weights are named

    monkeypatch.setattr(settings, "vision_embedder", "vgg16")
    assert configured_embedder_version() == Vgg16Embedder.model_version
    assert not is_omiclip_version(Vgg16Embedder.model_version)


def test_image_latent_round_trips_through_the_padded_embedding():
    latent = _unit(np.random.default_rng(0).normal(size=(1, OMICLIP_DIM)))[0]
    padded = np.concatenate([latent, np.zeros(EMBEDDING_DIM - OMICLIP_DIM)])

    np.testing.assert_array_equal(image_latent_from_embedding(padded), latent)


class _TinyOmiclip:
    """A small randomly initialized open_clip CoCa ViT standing in for
    OmiCLIP: same code paths (`encode_image`, transformer resblocks,
    attentional pooler), no 7.7 GB checkpoint.
    """

    def __init__(self):
        import open_clip

        torch.manual_seed(0)
        self._model = open_clip.CoCa(
            embed_dim=32,
            multimodal_cfg={"width": 32, "heads": 2, "layers": 1, "context_length": 16},
            text_cfg={"width": 32, "heads": 2, "layers": 1, "context_length": 16, "vocab_size": 64},
            vision_cfg={
                "image_size": 56, "patch_size": 14, "width": 32, "layers": 2, "head_width": 16,
                "attentional_pool": True, "attn_pooler_heads": 2, "output_tokens": True,
            },
        ).eval()

    def image_tensor(self, patches):
        tensor = torch.stack([torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1).float() / 255 for p in patches])
        return torch.nn.functional.interpolate(tensor, size=(56, 56), mode="bilinear")


def test_vit_gradcam_returns_a_normalized_map_at_patch_resolution():
    from nexus_agent.xai.gradcam import grad_cam_plusplus_vit

    embedder = _TinyOmiclip()
    patch = (np.random.default_rng(0).random((30, 22, 3)) * 255).astype(np.uint8)
    target = _unit(np.random.default_rng(1).normal(size=(1, 32)))[0]

    for heatmap in (grad_cam_plusplus_vit(embedder, patch), grad_cam_plusplus_vit(embedder, patch, target)):
        assert heatmap.shape == (30, 22)
        assert heatmap.dtype == np.float32
        assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
        assert np.isfinite(heatmap).all()
    assert all(p.grad is None for p in embedder._model.parameters())  # leaves no gradients behind


def test_vit_gradcam_depends_on_the_target():
    from nexus_agent.xai.gradcam import grad_cam_plusplus_vit

    embedder = _TinyOmiclip()
    patch = (np.random.default_rng(0).random((28, 28, 3)) * 255).astype(np.uint8)
    rng = np.random.default_rng(2)
    a, b = (_unit(rng.normal(size=(1, 32)))[0] for _ in range(2))

    assert not np.allclose(grad_cam_plusplus_vit(embedder, patch, a), grad_cam_plusplus_vit(embedder, patch, b))


# --- surfacing agreement --------------------------------------------------------


def test_report_shows_agreement_beside_the_claim_without_changing_its_tier():
    import uuid

    from nexus_agent.report.build import ReportClaimInput, render_report, render_report_html

    claims = [
        ReportClaimInput(claim_id="a", spatial_domain="domain_0", confidence=0.9, modality_agreement=0.31),
        ReportClaimInput(claim_id="b", spatial_domain="domain_0", confidence=0.9),
    ]
    report = render_report(uuid.uuid4(), uuid.uuid4(), claims)

    assert [e.modality_agreement for e in report.entries] == [0.31, None]
    assert report.entries[0].weight_tier == report.entries[1].weight_tier == "high"
    html_doc = render_report_html(report)
    assert html_doc.count("Image-expression agreement") == 1 and "0.31" in html_doc


def test_analyst_attaches_agreement_when_cells_were_embedded_with_omiclip(monkeypatch):
    from tests.test_analyst import _run_fused, _two_population_inputs  # noqa: F401  (fixture helpers)

    class _FakeTextEncoder:
        def text_latents(self, sentences):
            # profile i's latent = image i's latent for population-0 cells, the
            # opposite for population-1 cells: agreement high, then low.
            return _text_latents_for_test

    state, adata, population = _two_population_inputs()
    image_latents = _unit(np.random.default_rng(0).normal(size=(20, OMICLIP_DIM)))
    for cell, latent in zip(state["history"][0].payload["cells"], image_latents):
        cell["embedding_vector"] = np.concatenate([latent, np.zeros(EMBEDDING_DIM - OMICLIP_DIM)]).tolist()
        cell["embedding_model_version"] = OmiclipEmbedder.model_version
    global _text_latents_for_test
    _text_latents_for_test = np.where(population[:, None] == 0, image_latents, -image_latents)

    monkeypatch.setattr("nexus_agent.vision.omiclip.get_omiclip_embedder", lambda: _FakeTextEncoder())
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr("nexus_agent.learning.celltyping.latest_promoted_checkpoint_bytes", lambda dsn: None)
    from nexus_agent.agents.analyst import _ingest_and_fuse

    claims, _confidence, reasoning, _provenance = _ingest_and_fuse(state, "s3://fake/expr.h5ad", vision_ran=True)

    agreement = np.array([c.modality_agreement for c in claims])
    assert (agreement[population == 0] == 1.0).all()
    assert (agreement[population == 1] < 0.5).all()
    assert any("image-expression agreement" in line and "0.5 = chance" in line for line in reasoning)


def test_analyst_leaves_agreement_unset_for_other_encoders(monkeypatch):
    from tests.test_analyst import _run_fused

    claims, *_ = _run_fused(monkeypatch)  # fixture cells use embedding_model_version="test-stub"

    assert all(c.modality_agreement is None for c in claims)
