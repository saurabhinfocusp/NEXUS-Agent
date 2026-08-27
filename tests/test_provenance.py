"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Art. V §2's provenance pointer requirement is actually persisted
into the `provenance_log` table `db/schema.sql` creates (Phase 0).
"""

import uuid

import psycopg
import pytest

from nexus_agent.data.provenance import fetch_provenance, record_provenance
from nexus_agent.shared.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


def test_record_provenance_is_queryable_back():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()

    record_provenance(
        settings.postgres_dsn,
        run_id=run_id,
        task_id=task_id,
        claim_id="cell-1",
        source_image_region={"cell_id": "cell-1", "centroid_xy": [1.0, 2.0]},
        source_expression_profile={"obs_name": "cell_0"},
        component="analyst",
        component_version="0.1.0",
    )

    rows = fetch_provenance(settings.postgres_dsn, run_id)

    assert len(rows) == 1
    assert rows[0]["claim_id"] == "cell-1"
    assert rows[0]["source_image_region"] == {"cell_id": "cell-1", "centroid_xy": [1.0, 2.0]}
    assert rows[0]["source_expression_profile"] == {"obs_name": "cell_0"}
    assert rows[0]["component"] == "analyst"
    assert rows[0]["component_version"] == "0.1.0"
