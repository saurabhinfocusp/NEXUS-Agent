"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Art. VII §1's expert correction interface actually persists into
and is queryable back from the `correction_log` table `db/schema.sql`
creates (Phase 0, extended Phase 5).
"""

import uuid

import psycopg
import pytest

from nexus_agent.review.correction import (
    CorrectionRequest,
    fetch_corrections,
    submit_correction,
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


def test_submit_and_fetch_correction_round_trip():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()

    new_id = submit_correction(
        settings.postgres_dsn,
        CorrectionRequest(
            run_id=run_id,
            task_id=task_id,
            claim_id="cell-1",
            field="cell_type",
            reviewer="dr.jane",
            original_value={"cell_type": "T-cell"},
            corrected_value={"cell_type": "B-cell"},
            reason="marker mismatch",
            tissue_type="tonsil",
        ),
    )

    assert isinstance(new_id, int)

    rows = fetch_corrections(settings.postgres_dsn, run_id=run_id)

    assert len(rows) == 1
    row = rows[0]
    assert row.id == new_id
    assert row.run_id == run_id
    assert row.task_id == task_id
    assert row.claim_id == "cell-1"
    assert row.field == "cell_type"
    assert row.reviewer == "dr.jane"
    assert row.original_value == {"cell_type": "T-cell"}
    assert row.corrected_value == {"cell_type": "B-cell"}
    assert row.reason == "marker mismatch"
    assert row.tissue_type == "tonsil"


def test_fetch_corrections_filters_by_run_id_and_since():
    other_run_id = uuid.uuid4()
    task_id = uuid.uuid4()

    submit_correction(
        settings.postgres_dsn,
        CorrectionRequest(
            run_id=other_run_id,
            task_id=task_id,
            claim_id="cell-2",
            field="spatial_domain",
            reviewer="dr.jane",
            original_value={"domain": "stroma"},
            corrected_value={"domain": "tumor"},
        ),
    )

    # Filtering by run_id should only return rows for that run.
    rows = fetch_corrections(settings.postgres_dsn, run_id=other_run_id)
    assert len(rows) == 1
    assert rows[0].run_id == other_run_id

    # A `since` far in the future should exclude everything.
    from datetime import datetime, timedelta, timezone

    future = datetime.now(timezone.utc) + timedelta(days=1)
    rows_future = fetch_corrections(settings.postgres_dsn, run_id=other_run_id, since=future)
    assert rows_future == []

    # A `since` far in the past should include it.
    past = datetime.now(timezone.utc) - timedelta(days=1)
    rows_past = fetch_corrections(settings.postgres_dsn, run_id=other_run_id, since=past)
    assert len(rows_past) == 1
