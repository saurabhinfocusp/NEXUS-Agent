"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Art. XII §8's "veto rate and correction-rate trend are first-class
dashboards, not log-mining exercises... latency and confidence distribution
per agent" requirement is actually MEASURED from `message_log` /
`correction_log` (`db/schema.sql`), not scraped from checkpointer blobs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from nexus_agent.metrics.dashboards import (
    confidence_distribution,
    correction_rate_trend,
    latency_by_agent,
    prometheus_exposition,
    veto_rate,
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


def _insert_message(conn, *, run_id, task_id, from_agent, to_agent, verdict, confidence, created_at):
    conn.execute(
        """
        INSERT INTO message_log (run_id, task_id, from_agent, to_agent, verdict, confidence, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (run_id, task_id, from_agent, to_agent, verdict, confidence, created_at),
    )


def _insert_correction(conn, *, run_id, claim_id, reviewer, created_at):
    conn.execute(
        """
        INSERT INTO correction_log (run_id, claim_id, reviewer, original_value, corrected_value, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (run_id, claim_id, reviewer, '{"v": 1}', '{"v": 2}', created_at),
    )


@pytest.fixture
def dsn():
    return settings.postgres_dsn


@pytest.fixture(autouse=True)
def _isolate_tables(dsn):
    """Each test works with its own run_id/task_id namespace, but since the
    dashboard queries aggregate across the whole table, clear both tables
    before each test so hand-computed expectations hold exactly."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM message_log")
        conn.execute("DELETE FROM correction_log")
    yield


def test_veto_rate_matches_hand_calculation(dsn):
    run_id = uuid.uuid4()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with psycopg.connect(dsn, autocommit=True) as conn:
        # 4 critic verdicts: 1 veto, 2 pass, 1 escalate -> veto_rate = 1/4
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic",
            to_agent="orchestrator", verdict="veto", confidence=0.4, created_at=base,
        )
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic",
            to_agent="orchestrator", verdict="pass", confidence=0.9, created_at=base,
        )
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic",
            to_agent="orchestrator", verdict="pass", confidence=0.8, created_at=base,
        )
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic",
            to_agent="orchestrator", verdict="escalate", confidence=0.5, created_at=base,
        )
        # a non-critic message with a verdict should be ignored.
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="analyst",
            to_agent="critic", verdict="veto", confidence=0.9, created_at=base,
        )

    assert veto_rate(dsn) == pytest.approx(0.25)


def test_veto_rate_zero_total_returns_zero(dsn):
    assert veto_rate(dsn) == 0.0


def test_veto_rate_respects_since(dsn):
    run_id = uuid.uuid4()
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with psycopg.connect(dsn, autocommit=True) as conn:
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic",
            to_agent="orchestrator", verdict="veto", confidence=0.4, created_at=old,
        )
        _insert_message(
            conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic",
            to_agent="orchestrator", verdict="pass", confidence=0.9, created_at=new,
        )

    assert veto_rate(dsn, since=datetime(2025, 1, 1, tzinfo=timezone.utc)) == pytest.approx(0.0)
    assert veto_rate(dsn) == pytest.approx(0.5)


def test_correction_rate_trend_buckets_match_expected(dsn):
    run_id = uuid.uuid4()
    with psycopg.connect(dsn, autocommit=True) as conn:
        # week 1 (Jan 1-4 2026, a Thursday-Sunday): 2 corrections
        _insert_correction(conn, run_id=run_id, claim_id="c1", reviewer="alice", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        _insert_correction(conn, run_id=run_id, claim_id="c2", reviewer="alice", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
        # week 2 (Jan 8): 1 correction
        _insert_correction(conn, run_id=run_id, claim_id="c3", reviewer="bob", created_at=datetime(2026, 1, 8, tzinfo=timezone.utc))

    trend = correction_rate_trend(dsn, bucket="week")
    counts = [row["count"] for row in trend]
    assert sum(counts) == 3
    assert len(trend) == 2
    assert counts == [2, 1]


def test_correction_rate_trend_invalid_bucket_raises(dsn):
    with pytest.raises(ValueError):
        correction_rate_trend(dsn, bucket="fortnight")


def test_latency_by_agent_matches_hand_computed_deltas(dsn):
    run_id = uuid.uuid4()
    task_a = uuid.uuid4()
    task_b = uuid.uuid4()
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    with psycopg.connect(dsn, autocommit=True) as conn:
        # task_a: vision -> analyst (+10s) -> critic (+20s)
        _insert_message(conn, run_id=run_id, task_id=task_a, from_agent="vision", to_agent="analyst", verdict=None, confidence=0.9, created_at=base)
        _insert_message(conn, run_id=run_id, task_id=task_a, from_agent="analyst", to_agent="critic", verdict=None, confidence=0.8, created_at=base + timedelta(seconds=10))
        _insert_message(conn, run_id=run_id, task_id=task_a, from_agent="critic", to_agent="orchestrator", verdict="pass", confidence=0.7, created_at=base + timedelta(seconds=30))

        # task_b: vision -> analyst (+5s) -> critic (+40s)
        _insert_message(conn, run_id=run_id, task_id=task_b, from_agent="vision", to_agent="analyst", verdict=None, confidence=0.9, created_at=base)
        _insert_message(conn, run_id=run_id, task_id=task_b, from_agent="analyst", to_agent="critic", verdict=None, confidence=0.85, created_at=base + timedelta(seconds=5))
        _insert_message(conn, run_id=run_id, task_id=task_b, from_agent="critic", to_agent="orchestrator", verdict="pass", confidence=0.75, created_at=base + timedelta(seconds=45))

    stats = latency_by_agent(dsn)

    # "vision" never appears as curr (it's always first in a task), so it's
    # never a latency-attributed agent and must be omitted.
    assert "vision" not in stats

    # analyst latencies: 10s (task_a), 5s (task_b)
    assert stats["analyst"]["n"] == 2
    assert stats["analyst"]["mean_seconds"] == pytest.approx((10 + 5) / 2)
    assert stats["analyst"]["p50_seconds"] == pytest.approx(7.5)

    # critic latencies: 20s (task_a), 40s (task_b)
    assert stats["critic"]["n"] == 2
    assert stats["critic"]["mean_seconds"] == pytest.approx((20 + 40) / 2)
    assert stats["critic"]["p50_seconds"] == pytest.approx(30.0)


def test_confidence_distribution_matches_hand_computed_stats(dsn):
    run_id = uuid.uuid4()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with psycopg.connect(dsn, autocommit=True) as conn:
        for conf in (0.6, 0.8, 1.0):
            _insert_message(conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic", to_agent="orchestrator", verdict="pass", confidence=conf, created_at=base)
        for conf in (0.5, 0.9):
            _insert_message(conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="vision", to_agent="analyst", verdict=None, confidence=conf, created_at=base)

    dist = confidence_distribution(dsn)

    assert dist["critic"]["n"] == 3
    assert dist["critic"]["mean"] == pytest.approx(0.8)
    assert dist["critic"]["min"] == pytest.approx(0.6)
    assert dist["critic"]["max"] == pytest.approx(1.0)

    assert dist["vision"]["n"] == 2
    assert dist["vision"]["mean"] == pytest.approx(0.7)
    assert dist["vision"]["min"] == pytest.approx(0.5)
    assert dist["vision"]["max"] == pytest.approx(0.9)


def test_prometheus_exposition_contains_expected_metrics_and_does_not_crash_when_empty(dsn):
    # tables are empty at this point (autouse fixture cleared them).
    text = prometheus_exposition(dsn)
    assert "# HELP nexus_veto_rate" in text
    assert "nexus_veto_rate 0.0" in text
    assert "# HELP nexus_correction_count_total" in text
    assert "nexus_correction_count_total 0" in text
    assert "# HELP nexus_confidence_mean" in text
    assert "# HELP nexus_latency_mean_seconds" in text

    # now populate a bit and confirm per-agent lines show up.
    run_id = uuid.uuid4()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with psycopg.connect(dsn, autocommit=True) as conn:
        _insert_message(conn, run_id=run_id, task_id=uuid.uuid4(), from_agent="critic", to_agent="orchestrator", verdict="veto", confidence=0.83, created_at=base)
        _insert_correction(conn, run_id=run_id, claim_id="c1", reviewer="alice", created_at=base)

    text = prometheus_exposition(dsn)
    assert 'nexus_confidence_mean{agent="critic"}' in text
    assert "nexus_correction_count_total 1" in text
