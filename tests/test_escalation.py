"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Art. VII §3's reviewer-facing escalation queue actually persists
into, lists from, and resolves rows in the `escalation_queue` table
`db/schema.sql` creates (Phase 5).
"""

import uuid

import psycopg
import pytest

from nexus_agent.review.escalation import (
    list_pending_escalations,
    record_escalation,
    resolve_escalation,
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


def test_record_list_and_resolve_escalation_round_trip():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()

    new_id = record_escalation(
        settings.postgres_dsn,
        run_id=run_id,
        task_id=task_id,
        claim_id="cell-1",
        payload={"reason": "low confidence", "verdict": "uncertain"},
        confidence=0.32,
    )

    assert isinstance(new_id, int)

    pending = list_pending_escalations(settings.postgres_dsn)
    matching = [r for r in pending if r.id == new_id]
    assert len(matching) == 1
    record = matching[0]
    assert record.run_id == run_id
    assert record.task_id == task_id
    assert record.claim_id == "cell-1"
    assert record.payload == {"reason": "low confidence", "verdict": "uncertain"}
    assert record.confidence == pytest.approx(0.32)
    assert record.status == "pending"
    assert record.resolved_at is None

    resolve_escalation(settings.postgres_dsn, new_id)

    pending_after = list_pending_escalations(settings.postgres_dsn)
    assert new_id not in {r.id for r in pending_after}
