"""Tests for the reviewer-facing FastAPI surface (Constitution Art. VII
§1, §3): src/nexus_agent/review/api.py.

The first test is fast and requires no infra -- it just verifies the app
is constructible and routed correctly. The rest are integration tests
against a real Postgres instance, using the same skip-if-unreachable
fixture as tests/test_provenance.py -- requested explicitly (not
module-autouse) so the fast test above is never gated on Postgres.
"""

import uuid

import psycopg
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from nexus_agent.review.api import app
from nexus_agent.review.escalation import record_escalation
from nexus_agent.shared.config import settings


def test_app_routes_exist():
    paths = {r.path for r in app.routes if isinstance(r, APIRoute)}
    assert "/escalations" in paths
    assert "/escalations/{escalation_id}/resolve" in paths
    assert "/corrections" in paths
    assert "/metrics" in paths


@pytest.fixture()
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.mark.integration
def test_post_and_get_corrections_round_trip(_require_postgres, client):
    run_id, task_id = str(uuid.uuid4()), str(uuid.uuid4())

    response = client.post(
        "/corrections",
        json={
            "run_id": run_id,
            "task_id": task_id,
            "claim_id": "cell-42",
            "field": "interpretation",
            "reviewer": "dr.jane",
            "original_value": {"note": "benign"},
            "corrected_value": {"note": "malignant"},
            "reason": "re-review",
            "tissue_type": "lung",
        },
    )
    assert response.status_code == 201
    new_id = response.json()["id"]
    assert isinstance(new_id, int)

    get_response = client.get("/corrections", params={"run_id": run_id})
    assert get_response.status_code == 200
    rows = get_response.json()
    assert len(rows) == 1
    assert rows[0]["id"] == new_id
    assert rows[0]["claim_id"] == "cell-42"
    assert rows[0]["corrected_value"] == {"note": "malignant"}


@pytest.mark.integration
def test_escalation_list_and_resolve_round_trip(_require_postgres, client):
    run_id, task_id = uuid.uuid4(), uuid.uuid4()

    new_id = record_escalation(
        settings.postgres_dsn,
        run_id=run_id,
        task_id=task_id,
        claim_id="cell-99",
        payload={"reason": "conflicting verdicts"},
        confidence=0.4,
    )

    list_response = client.get("/escalations")
    assert list_response.status_code == 200
    ids = {row["id"] for row in list_response.json()}
    assert new_id in ids

    resolve_response = client.post(f"/escalations/{new_id}/resolve")
    assert resolve_response.status_code == 200
    assert resolve_response.json() == {"status": "resolved", "id": new_id}

    list_after = client.get("/escalations")
    assert list_after.status_code == 200
    ids_after = {row["id"] for row in list_after.json()}
    assert new_id not in ids_after


@pytest.mark.integration
def test_metrics_endpoint_is_well_formed_either_way(_require_postgres, client):
    response = client.get("/metrics")
    assert response.status_code in (200, 503)
    assert isinstance(response.text, str)
    if response.status_code == 503:
        assert "metrics module not yet available" in response.text
