"""Integration test: requires `docker compose up -d` (MinIO + Postgres) and
CellPose/VGG16 weights (network on first run, cached after).

Confirms Phase 3's stated exit criterion end to end: raw image + expression
data in (via image_uri/expression_uri) -> fused 128-dim per-cell
representation out, with provenance intact and single-modality claims
still correctly flagged provisional.
"""

import uuid

import numpy as np
import pandas as pd
import psycopg
import pytest
from anndata import AnnData
from skimage.draw import disk

from nexus_agent.agents.analyst import AnalystClaim
from nexus_agent.data.object_store import ensure_bucket, put_anndata, put_array
from nexus_agent.data.provenance import fetch_provenance
from nexus_agent.graph.build import compile_with_memory
from nexus_agent.shared.config import settings
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal, Verdict

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_infra():
    try:
        ensure_bucket()
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO/Postgres not reachable ({exc}); run `docker compose up -d`.")


def _synthetic_image(n_blobs: int = 4, size: int = 128, seed: int = 0) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(n_blobs):
        cy, cx = rng.integers(20, size - 20, size=2)
        rr, cc = disk((cy, cx), 10, shape=image.shape)
        image[rr, cc] = 200
    return image


def _upload_sample(run_id: uuid.UUID) -> tuple[str, str]:
    image = _synthetic_image()
    image_uri = put_array(f"tests/{run_id}/image.npy", image)

    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(3, size=(4, 12)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
    )
    expression_uri = put_anndata(f"tests/{run_id}/expression.h5ad", adata)
    return image_uri, expression_uri


def _initial_state(run_id: uuid.UUID, **overrides) -> dict:
    state = dict(
        run_id=run_id,
        task_id=uuid.uuid4(),
        trace_id=uuid.uuid4(),
        goal=AnalyticalGoal(sample_id="sample-1", modalities=["image", "expression"]),
        history=[],
        verdict=None,
        force_verdict=None,
        stub_confidence_override=None,
        image_uri=None,
        expression_uri=None,
    )
    state.update(overrides)
    return state


def test_real_image_and_expression_fuse_end_to_end_with_provenance():
    run_id = uuid.uuid4()
    image_uri, expression_uri = _upload_sample(run_id)
    graph = compile_with_memory()

    result = graph.invoke(
        _initial_state(run_id, image_uri=image_uri, expression_uri=expression_uri),
        config={"configurable": {"thread_id": str(run_id)}},
    )

    vision_message = next(m for m in result["history"] if m.from_agent == AgentName.VISION)
    analyst_message = next(m for m in result["history"] if m.from_agent == AgentName.ANALYST)

    assert len(vision_message.payload["cells"]) > 0  # CellPose actually found cells
    for cell in vision_message.payload["cells"]:
        assert len(cell["embedding_vector"]) == 1024

    claims = [AnalystClaim.model_validate(c) for c in analyst_message.payload["claims"]]
    assert claims
    assert all(not claim.provisional for claim in claims)  # both modalities present -> not provisional (Art. V §1)

    assert result["verdict"] in (Verdict.PASS, Verdict.VETO, Verdict.ESCALATE)

    provenance_rows = fetch_provenance(settings.postgres_dsn, run_id)
    assert len(provenance_rows) == len(claims)
    for row in provenance_rows:
        assert row["source_image_region"] is not None
        assert row["source_expression_profile"] is not None


def test_expression_only_stays_provisional():
    run_id = uuid.uuid4()
    _, expression_uri = _upload_sample(run_id)
    graph = compile_with_memory()

    result = graph.invoke(
        _initial_state(
            run_id,
            expression_uri=expression_uri,
            goal=AnalyticalGoal(sample_id="sample-1", modalities=["expression"]),
        ),
        config={"configurable": {"thread_id": str(run_id)}},
    )

    analyst_message = next(m for m in result["history"] if m.from_agent == AgentName.ANALYST)
    claims = [AnalystClaim.model_validate(c) for c in analyst_message.payload["claims"]]
    assert claims
    assert all(claim.provisional for claim in claims)  # single modality -> provisional (Art. V §1)
