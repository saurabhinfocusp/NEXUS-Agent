"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Art. VI §3's auditability requirement -- "a domain expert or
auditor can verify a claim without re-running the pipeline" -- by round-
tripping a `ClaimEvidenceBundle` through `xai_evidence` and fetching it back
using only `dsn, run_id, claim_id`, with no report-generation code involved.
"""

import uuid

import numpy as np
import psycopg
import pytest

from nexus_agent.xai.claims_evidence import (
    ClaimEvidenceBundle,
    fetch_evidence_bundle,
    store_evidence_bundle,
)
from nexus_agent.shared.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


def test_store_and_fetch_evidence_bundle_round_trips():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    heatmap = np.random.default_rng(0).random((8, 8)).astype(np.float32)

    bundle = ClaimEvidenceBundle(
        claim_id="cell-1",
        claim_type="cell_type_call",
        shap_top_genes=[{"gene": "CD3E", "shap_value": 0.42}],
        citations=[{"source": "PMID:12345", "similarity": 0.81}],
        no_literature_retrieved=False,
        confidence=0.9,
        artifacts_expected=["heatmap", "shap_top_genes", "citations"],
        artifacts_present=["heatmap", "shap_top_genes", "citations"],
        component_version="grad-cam++-vgg16-v1",
    )

    stored = store_evidence_bundle(settings.postgres_dsn, run_id, task_id, bundle, heatmap_array=heatmap)

    assert stored.heatmap_uri is not None
    assert stored.heatmap_uri.startswith("s3://")

    fetched = fetch_evidence_bundle(settings.postgres_dsn, run_id, "cell-1")

    assert fetched is not None
    assert fetched.claim_id == "cell-1"
    assert fetched.claim_type == "cell_type_call"
    assert fetched.heatmap_uri == stored.heatmap_uri
    assert fetched.shap_top_genes == [{"gene": "CD3E", "shap_value": 0.42}]
    assert fetched.citations == [{"source": "PMID:12345", "similarity": 0.81}]
    assert fetched.no_literature_retrieved is False
    assert fetched.confidence == 0.9
    assert fetched.artifacts_expected == ["heatmap", "shap_top_genes", "citations"]
    assert fetched.artifacts_present == ["heatmap", "shap_top_genes", "citations"]
    assert fetched.component_version == "grad-cam++-vgg16-v1"


def test_fetch_evidence_bundle_returns_none_when_missing():
    assert fetch_evidence_bundle(settings.postgres_dsn, uuid.uuid4(), "no-such-claim") is None
