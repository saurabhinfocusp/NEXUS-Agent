"""Queryable dashboard metrics (Constitution Art. XII §8).

"Veto rate and correction-rate trend are first-class dashboards, not
log-mining exercises... latency and confidence distribution per agent" --
every function here is MEASURED from `message_log` / `correction_log`
(`db/schema.sql`), populated once per emitted `MessageEnvelope` by a wrapper
in `graph/build.py`, never by scraping checkpointer blobs.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from itertools import groupby
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row


def veto_rate(dsn: str, *, since: datetime | None = None) -> float:
    """Fraction of Critic verdicts (`from_agent = 'critic'`) that are veto."""
    query = (
        "SELECT count(*) FILTER (WHERE verdict = 'veto') AS vetoes, "
        "count(*) AS total FROM message_log "
        "WHERE from_agent = 'critic' AND verdict IS NOT NULL"
    )
    params: tuple[Any, ...] = ()
    if since is not None:
        query += " AND created_at >= %s"
        params = (since,)

    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        row = conn.execute(query, params).fetchone()

    total = row["total"] if row else 0
    if not total:
        return 0.0
    return row["vetoes"] / total


def correction_rate_trend(dsn: str, *, bucket: str = "week") -> list[dict]:
    """Correction counts bucketed by day/week/month, for a trend chart."""
    if bucket not in {"day", "week", "month"}:
        raise ValueError(f"bucket must be one of 'day', 'week', 'month'; got {bucket!r}")

    # Safe to f-string interpolate: `bucket` is checked against a fixed
    # allowlist above, and date_trunc's unit argument can't be a normal bind
    # parameter across all query styles.
    query = (
        f"SELECT date_trunc('{bucket}', created_at) AS bucket, count(*) AS n "
        "FROM correction_log GROUP BY 1 ORDER BY 1"
    )

    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        rows = conn.execute(query).fetchall()

    return [{"bucket": row["bucket"], "count": row["n"]} for row in rows]


def latency_by_agent(dsn: str) -> dict[str, dict]:
    """Per-agent latency stats, computed from consecutive message timestamps
    within each `task_id`."""
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT run_id, task_id, from_agent, created_at FROM message_log "
            "ORDER BY task_id, created_at"
        ).fetchall()

    latencies_by_agent: dict[str, list[float]] = {}
    for _task_id, group in groupby(rows, key=lambda r: r["task_id"]):
        task_rows = list(group)
        for prev, curr in zip(task_rows, task_rows[1:]):
            delta = (curr["created_at"] - prev["created_at"]).total_seconds()
            latencies_by_agent.setdefault(curr["from_agent"], []).append(delta)

    result: dict[str, dict] = {}
    for agent, latencies in latencies_by_agent.items():
        if not latencies:
            continue
        arr = np.asarray(latencies, dtype=float)
        result[agent] = {
            "mean_seconds": float(statistics.mean(latencies)),
            "p50_seconds": float(np.percentile(arr, 50)),
            "p95_seconds": float(np.percentile(arr, 95)),
            "n": len(latencies),
        }
    return result


def confidence_distribution(dsn: str) -> dict[str, dict]:
    """Per-agent confidence distribution summary (mean/min/max/n)."""
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        rows = conn.execute("SELECT from_agent, confidence FROM message_log").fetchall()

    confidences_by_agent: dict[str, list[float]] = {}
    for row in rows:
        confidences_by_agent.setdefault(row["from_agent"], []).append(row["confidence"])

    result: dict[str, dict] = {}
    for agent, confidences in confidences_by_agent.items():
        if not confidences:
            continue
        result[agent] = {
            "mean": float(statistics.mean(confidences)),
            "min": float(min(confidences)),
            "max": float(max(confidences)),
            "n": len(confidences),
        }
    return result


def _correction_count_total(dsn: str) -> int:
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        row = conn.execute("SELECT count(*) AS n FROM correction_log").fetchone()
    return int(row["n"]) if row else 0


def prometheus_exposition(dsn: str) -> str:
    """Combine all dashboard metrics into Prometheus text exposition format.

    Must never raise, even against empty tables -- callers (e.g.
    `review/api.py`'s `/metrics` endpoint) expose this directly to scrapers.
    """
    lines: list[str] = []

    lines.append("# HELP nexus_veto_rate Fraction of Critic verdicts that are veto")
    lines.append("# TYPE nexus_veto_rate gauge")
    lines.append(f"nexus_veto_rate {veto_rate(dsn)}")

    lines.append("# HELP nexus_correction_count_total Total corrections logged")
    lines.append("# TYPE nexus_correction_count_total counter")
    lines.append(f"nexus_correction_count_total {_correction_count_total(dsn)}")

    lines.append("# HELP nexus_confidence_mean Mean confidence per agent")
    lines.append("# TYPE nexus_confidence_mean gauge")
    for agent, stats in confidence_distribution(dsn).items():
        lines.append(f'nexus_confidence_mean{{agent="{agent}"}} {stats["mean"]}')

    latency_stats = latency_by_agent(dsn)

    lines.append("# HELP nexus_latency_mean_seconds Mean latency per agent, seconds")
    lines.append("# TYPE nexus_latency_mean_seconds gauge")
    for agent, stats in latency_stats.items():
        lines.append(f'nexus_latency_mean_seconds{{agent="{agent}"}} {stats["mean_seconds"]}')

    lines.append("# HELP nexus_latency_p95_seconds P95 latency per agent, seconds")
    lines.append("# TYPE nexus_latency_p95_seconds gauge")
    for agent, stats in latency_stats.items():
        lines.append(f'nexus_latency_p95_seconds{{agent="{agent}"}} {stats["p95_seconds"]}')

    return "\n".join(lines) + "\n"
