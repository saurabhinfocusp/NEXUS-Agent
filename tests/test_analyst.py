"""Fast/Docker-free unit tests for `agents/analyst.py`'s real-path ingestion.

Vision's segmented-cell count and an expression file's own spot/cell count
come from independent modalities and essentially never coincide unless a
test fixture deliberately constructs them to match (as
`tests/test_graph_real_pipeline.py`'s synthetic fixtures do) or the
expression file already carries its own `obsm["spatial"]`. When neither
holds, `_ingest_and_fuse` must fail with a clear, actionable error instead
of crashing inside AnnData's own obsm-shape check.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from nexus_agent.agents.analyst import _ingest_and_fuse
from nexus_agent.agents.common import build_envelope
from nexus_agent.shared.config import settings
from nexus_agent.shared.schemas import AgentName


@pytest.fixture(autouse=True)
def _no_promoted_checkpoint(monkeypatch):
    """The fused path looks up the latest promoted cell-type checkpoint in
    Postgres; these tests must run without it (CI has no database). Tests
    that want a checkpoint override this through `_run_fused`.
    """
    monkeypatch.setattr("nexus_agent.learning.celltyping.latest_promoted_checkpoint_bytes", lambda dsn: None)


def _state_with_vision_cells(n_cells: int) -> dict:
    state = {
        "run_id": uuid.uuid4(),
        "task_id": uuid.uuid4(),
        "trace_id": uuid.uuid4(),
        "history": [],
    }
    cells = [
        {
            "cell_id": f"cell-{i}",
            "centroid_xy": (float(i), float(i)),
            "mask_polygon": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            "embedding_vector": [0.0] * 1024,
            "embedding_model_version": "test-stub",
        }
        for i in range(n_cells)
    ]
    envelope = build_envelope(
        state,
        from_agent=AgentName.VISION,
        to_agent=AgentName.ANALYST,
        payload={"cells": cells},
        confidence=0.8,
    )
    state["history"] = [envelope]
    return state


def _adata(n_obs: int, *, with_spatial: bool = False) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(3, size=(n_obs, 12)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
    )
    if with_spatial:
        adata.obsm["spatial"] = rng.random((n_obs, 2)) * 100
    return adata


def test_mismatched_vision_and_expression_counts_raise_clear_error(monkeypatch):
    """4 Vision cells, 6 expression rows, expression has no obsm['spatial']
    -- must fail with an actionable message, not an AnnData internal one.
    """
    state = _state_with_vision_cells(4)
    monkeypatch.setattr(
        "nexus_agent.data.object_store.get_anndata", lambda uri: _adata(6, with_spatial=False)
    )

    with pytest.raises(ValueError, match="don't match Vision's"):
        _ingest_and_fuse(state, "s3://fake-bucket/expr.h5ad", vision_ran=True)


def test_matching_vision_and_expression_counts_still_fuse(monkeypatch):
    """The count-matched case (what the integration test fixtures rely on)
    must keep working unchanged.
    """
    state = _state_with_vision_cells(4)
    monkeypatch.setattr(
        "nexus_agent.data.object_store.get_anndata", lambda uri: _adata(4, with_spatial=False)
    )

    claims, confidence, reasoning, provenance_records = _ingest_and_fuse(
        state, "s3://fake-bucket/expr.h5ad", vision_ran=True
    )

    assert len(claims) == 4
    assert all(not claim.provisional for claim in claims)
    assert len(provenance_records) == 4


def test_expression_with_its_own_spatial_coords_ignores_vision_count(monkeypatch):
    """A real spatially-resolved platform (Visium/MERFISH/Xenium) already
    carries obsm['spatial'] -- Vision's cell count shouldn't matter at all.
    """
    state = _state_with_vision_cells(4)
    monkeypatch.setattr(
        "nexus_agent.data.object_store.get_anndata", lambda uri: _adata(6, with_spatial=True)
    )

    with pytest.raises(ValueError, match="1:1 aligned"):
        # Still correctly reaches fuse()'s own alignment check (a different,
        # legitimate mismatch: vision_cells vs. adata rows for the fusion
        # step itself), proving the obsm['spatial']-derivation branch is
        # skipped rather than raising the ingestion-side error above.
        _ingest_and_fuse(state, "s3://fake-bucket/expr.h5ad", vision_ran=True)


# ---------------------------------------------------------------------------
# The fused path: domains, measured confidence, and the fusion-base guard.
# Postgres (the promoted-checkpoint lookup) is monkeypatched.
# ---------------------------------------------------------------------------


def _two_population_inputs(n_per_region: int = 10):
    """Vision cells + expression for two populations in two regions, each
    with its own morphology signature and expression program.
    """
    rng = np.random.default_rng(0)
    n = 2 * n_per_region
    population = np.array([0] * n_per_region + [1] * n_per_region)
    counts = rng.poisson(5, size=(n, 20)).astype(np.float32)
    counts[population == 0, :10] += 60
    counts[population == 1, 10:] += 60
    adata = AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(20)]),
    )
    adata.obsm["spatial"] = rng.random((n, 2)) * 10 + population[:, None] * 1000

    state = _state_with_vision_cells(n)
    for cell, pop in zip(state["history"][0].payload["cells"], population):
        embedding = rng.normal(0.0, 0.05, size=1024)
        embedding[pop] += 3.0
        cell["embedding_vector"] = embedding.tolist()
    return state, adata, population


def _run_fused(monkeypatch, *, checkpoint_bytes=None):
    state, adata, population = _two_population_inputs()
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr(
        "nexus_agent.learning.celltyping.latest_promoted_checkpoint_bytes", lambda dsn: checkpoint_bytes
    )
    claims, confidence, reasoning, provenance = _ingest_and_fuse(state, "s3://fake/expr.h5ad", vision_ran=True)
    return claims, confidence, reasoning, provenance, population


def test_fused_claims_carry_a_spatial_domain_and_a_measured_confidence(monkeypatch):
    claims, confidence, reasoning, _provenance, population = _run_fused(monkeypatch)

    domains = [c.spatial_domain for c in claims]
    assert all(d is not None for d in domains)
    # each population lands in one domain, and the two differ
    assert len(set(domains[:10])) == 1 and len(set(domains[10:])) == 1 and domains[0] != domains[-1]
    assert all(c.cell_type is None for c in claims)  # no promoted checkpoint -> no call
    assert all(not c.provisional for c in claims)

    per_claim = [c.confidence for c in claims]
    assert len(set(per_claim)) > 1  # measured per cell, not one constant
    assert confidence == pytest.approx(np.mean(per_claim))
    assert any("silhouette" in line for line in reasoning)
    assert any("untrained fixed-seed weights" in line for line in reasoning)


def test_provenance_records_the_fusion_base_each_embedding_came_from(monkeypatch):
    from nexus_agent.analyst.pretrain import FIXED_SEED_BASE_ID, fusion_space_id

    _claims, _confidence, _reasoning, provenance, _population = _run_fused(monkeypatch)

    assert len(provenance) == 20
    expected = fusion_space_id(FIXED_SEED_BASE_ID, "test-stub")  # the fixture's embedding_model_version
    assert all(p["source_expression_profile"]["fusion_space_id"] == expected for p in provenance)


def _checkpoint(fusion_space_id: str) -> bytes:
    from nexus_agent.learning.celltyping import CellTypeClassifier, save_checkpoint

    return save_checkpoint(
        CellTypeClassifier(num_classes=2), {0: "T cell", 1: "B cell"}, None, fusion_space_id=fusion_space_id
    )


def test_checkpoint_from_the_active_base_is_applied_and_bounds_confidence(monkeypatch):
    from nexus_agent.analyst.pretrain import FIXED_SEED_BASE_ID, fusion_space_id

    untyped, *_ = _run_fused(monkeypatch)
    active_space = fusion_space_id(FIXED_SEED_BASE_ID, "test-stub")
    typed, _confidence, reasoning, *_ = _run_fused(monkeypatch, checkpoint_bytes=_checkpoint(active_space))

    assert all(c.cell_type in {"T cell", "B cell"} for c in typed)
    # a claim is as confident as its weakest part: adding a call can only lower it
    assert all(t.confidence <= u.confidence + 1e-9 for t, u in zip(typed, untyped))
    assert any("softmax probability" in line for line in reasoning)


@pytest.mark.parametrize(
    "other_space",
    [
        "pretrained-other|test-stub",  # another fusion base
        "fixed-seed-20260827|vgg16-conv5_3-zeropad1024-v1",  # same base, another morphology encoder
    ],
)
def test_checkpoint_from_another_fusion_space_is_not_applied_and_the_reason_is_recorded(monkeypatch, other_space):
    claims, _confidence, reasoning, *_ = _run_fused(monkeypatch, checkpoint_bytes=_checkpoint(other_space))

    assert all(c.cell_type is None for c in claims)
    assert any(other_space in line and "not applied" in line for line in reasoning)


def test_configured_pretrained_base_is_used_with_its_own_projector(monkeypatch):
    from nexus_agent.agents import analyst
    from nexus_agent.analyst.ingestion import normalize, spatial_knn_graph
    from nexus_agent.analyst.pretrain import pretrain_fusion, save_fusion_base

    _state, training_adata, _population = _two_population_inputs()
    training_adata = training_adata.copy()
    normalize(training_adata)
    spatial_knn_graph(training_adata)
    morphology = np.array([c["embedding_vector"] for c in _state["history"][0].payload["cells"]])
    base, _losses = pretrain_fusion(
        [(morphology, training_adata)], epochs=5, seed=0, morphology_model_version="test-stub"
    )

    monkeypatch.setattr(settings, "fusion_checkpoint_uri", "s3://fake/fusion/base.pt")
    monkeypatch.setattr("nexus_agent.data.object_store.get_bytes", lambda uri: save_fusion_base(base))
    analyst._load_pretrained_base.cache_clear()
    try:
        claims, _confidence, reasoning, provenance, _population = _run_fused(monkeypatch)
    finally:
        analyst._load_pretrained_base.cache_clear()

    assert len(claims) == 20
    assert all(
        p["source_expression_profile"]["fusion_space_id"] == f"{base.base_id}|test-stub" for p in provenance
    )
    assert any(base.base_id in line for line in reasoning)
    assert any("100% of the pretrained base's 20 genes" in line for line in reasoning)


def test_pretrained_base_refuses_embeddings_from_another_encoder(monkeypatch):
    from nexus_agent.agents import analyst
    from nexus_agent.analyst.ingestion import normalize, spatial_knn_graph
    from nexus_agent.analyst.pretrain import pretrain_fusion, save_fusion_base

    state, training_adata, _population = _two_population_inputs()
    training_adata = training_adata.copy()
    normalize(training_adata)
    spatial_knn_graph(training_adata)
    morphology = np.array([c["embedding_vector"] for c in state["history"][0].payload["cells"]])
    base, _losses = pretrain_fusion(
        [(morphology, training_adata)], epochs=2, seed=0, morphology_model_version="vgg16-conv5_3-zeropad1024-v1"
    )
    monkeypatch.setattr(settings, "fusion_checkpoint_uri", "s3://fake/fusion/base.pt")
    monkeypatch.setattr("nexus_agent.data.object_store.get_bytes", lambda uri: save_fusion_base(base))
    analyst._load_pretrained_base.cache_clear()
    try:
        with pytest.raises(ValueError, match="trained on vgg16"):
            _run_fused(monkeypatch)  # the fixture's cells say embedding_model_version="test-stub"
    finally:
        analyst._load_pretrained_base.cache_clear()
