"""Integration test: requires `docker compose up -d` (Postgres reachable).

Round-trips webapp/runs.py's pipeline_runs writers/reader (create -> running
-> done / failed).
"""

import uuid

import psycopg
import pytest

from nexus_agent.shared.config import settings
from nexus_agent.webapp.runs import create_run, fetch_run, mark_done, mark_failed, mark_running

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


def _new_run() -> tuple[uuid.UUID, uuid.UUID]:
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    create_run(
        settings.postgres_dsn,
        run_id=run_id,
        task_id=task_id,
        sample_id="sample-1",
        image_uri="s3://nexus-agent/webapp/test/image.npy",
        expression_uri="s3://nexus-agent/webapp/test/expression.h5ad",
    )
    return run_id, task_id


def test_create_run_starts_pending():
    run_id, task_id = _new_run()

    run = fetch_run(settings.postgres_dsn, run_id)

    assert run is not None
    assert run.status == "pending"
    assert run.task_id == task_id
    assert run.verdict is None
    assert run.claims is None


def test_mark_running_then_done_is_queryable_back():
    run_id, _ = _new_run()

    mark_running(settings.postgres_dsn, run_id)
    assert fetch_run(settings.postgres_dsn, run_id).status == "running"

    claims = [{"cell_id": "cell-1", "cell_type": "T-cell", "confidence": 0.8}]
    mark_done(settings.postgres_dsn, run_id, verdict="pass", claims=claims, report_html="<html></html>")

    run = fetch_run(settings.postgres_dsn, run_id)
    assert run.status == "done"
    assert run.verdict == "pass"
    assert run.claims == claims
    assert run.report_html == "<html></html>"
    assert run.completed_at is not None


def test_mark_failed_records_the_error():
    run_id, _ = _new_run()

    mark_failed(settings.postgres_dsn, run_id, "boom")

    run = fetch_run(settings.postgres_dsn, run_id)
    assert run.status == "failed"
    assert run.error == "boom"
    assert run.completed_at is not None


def test_fetch_run_returns_none_for_unknown_id():
    assert fetch_run(settings.postgres_dsn, uuid.uuid4()) is None
