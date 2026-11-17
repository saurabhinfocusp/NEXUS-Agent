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
from nexus_agent.shared.schemas import AgentName


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
