"""Unit-level tests for the Spatial Transcriptomics agent (Art. III §4,
Art. V §1) -- calls `spatial_node` directly with a hand-built state,
mirroring `test_critic.py`'s style; the real path is exercised by
monkeypatching `data.object_store.get_anndata` with synthetic AnnData so
these stay Docker-free (same convention as `tests/test_qc.py`).
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from anndata import AnnData

from nexus_agent.agents.spatial import SpatialClaim, spatial_node
from nexus_agent.shared.schemas import AgentName


def _base_state(**overrides):
    run_id, task_id, trace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    state = dict(
        run_id=run_id,
        task_id=task_id,
        trace_id=trace_id,
        history=[],
        subtask_plan=[AgentName.ANALYST, AgentName.SPATIAL],
        verdict=None,
        force_verdict=None,
        stub_confidence_override=None,
        image_uri=None,
        expression_uri=None,
    )
    state.update(overrides)
    return state


def test_stub_path_produces_one_synthetic_claim_and_targets_critic():
    state = _base_state()
    result = spatial_node(state)

    envelope = result["history"][0]
    assert envelope.from_agent == AgentName.SPATIAL
    assert envelope.to_agent == AgentName.CRITIC
    claims = [SpatialClaim.model_validate(c) for c in envelope.payload["claims"]]
    assert len(claims) == 1
    assert 0.0 <= envelope.confidence <= 1.0


def test_stub_confidence_override_is_honored():
    state = _base_state(stub_confidence_override={AgentName.SPATIAL: 0.05})
    result = spatial_node(state)
    assert result["history"][0].confidence == 0.05


def test_retry_after_veto_reports_higher_confidence():
    from nexus_agent.shared.schemas import MessageEnvelope, Verdict

    run_id, task_id, trace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    veto_message = MessageEnvelope(
        run_id=run_id,
        task_id=task_id,
        from_agent=AgentName.CRITIC,
        to_agent=AgentName.SPATIAL,
        payload={"verdict": Verdict.VETO.value, "objection": "low confidence"},
        confidence=0.9,
        trace_id=trace_id,
    )
    state = _base_state(run_id=run_id, task_id=task_id, trace_id=trace_id, history=[veto_message])

    result = spatial_node(state)
    assert result["history"][0].confidence > 0.5  # acted on the objection, not just received it


def _synthetic_adata(n_cells: int = 12, n_genes: int = 10) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
    )
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    return adata


def test_real_path_computes_niche_claims_per_cell(monkeypatch):
    adata = _synthetic_adata()
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)

    state = _base_state(expression_uri="s3://fake-bucket/expr.h5ad")
    result = spatial_node(state)

    envelope = result["history"][0]
    claims = [SpatialClaim.model_validate(c) for c in envelope.payload["claims"]]
    assert len(claims) == adata.n_obs
    assert {c.cell_id for c in claims} == set(adata.obs_names)
    assert all(c.niche_label.startswith("niche_") for c in claims)
    assert all("nhood_enrichment_zscore" in c.supporting_stat for c in claims)
    assert any("squidpy.gr.spatial_autocorr" in line for line in envelope.payload["reasoning"])


def test_real_path_routes_to_biology_when_in_plan(monkeypatch):
    adata = _synthetic_adata()
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)

    state = _base_state(
        expression_uri="s3://fake-bucket/expr.h5ad",
        subtask_plan=[AgentName.ANALYST, AgentName.SPATIAL, AgentName.BIOLOGY],
    )
    result = spatial_node(state)
    assert result["history"][0].to_agent == AgentName.BIOLOGY


def test_real_path_derives_spatial_coords_from_vision_when_missing(monkeypatch):
    n_cells = 5
    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(5, size=(n_cells, 6)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
    )
    assert "spatial" not in adata.obsm
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)

    from nexus_agent.shared.schemas import MessageEnvelope

    run_id, task_id, trace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    vision_message = MessageEnvelope(
        run_id=run_id,
        task_id=task_id,
        from_agent=AgentName.VISION,
        to_agent=AgentName.ANALYST,
        payload={"cells": [{"centroid_xy": (float(i), float(i)), "cell_id": f"vc{i}"} for i in range(n_cells)]},
        confidence=0.8,
        trace_id=trace_id,
    )
    state = _base_state(
        run_id=run_id,
        task_id=task_id,
        trace_id=trace_id,
        history=[vision_message],
        expression_uri="s3://fake-bucket/expr.h5ad",
        subtask_plan=[AgentName.VISION, AgentName.ANALYST, AgentName.SPATIAL],
    )

    result = spatial_node(state)
    claims = result["history"][0].payload["claims"]
    assert len(claims) == n_cells
