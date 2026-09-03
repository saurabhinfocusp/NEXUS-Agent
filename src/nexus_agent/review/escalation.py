"""Reviewer-facing escalation queue (Constitution Art. VII §3).

"The human-review queue node from Phase 1/2 needs an actual reviewer-facing
surface now, not just a terminating graph edge." Writes into and reads from
the `escalation_queue` table `db/schema.sql` already creates.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import psycopg
from pydantic import BaseModel


class EscalationRecord(BaseModel):
    id: int
    run_id: uuid.UUID
    task_id: uuid.UUID
    claim_id: str | None = None
    payload: dict[str, Any]
    confidence: float
    status: str
    created_at: datetime
    resolved_at: datetime | None = None


def record_escalation(
    dsn: str,
    *,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    claim_id: str | None,
    payload: dict[str, Any],
    confidence: float,
) -> int:
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            """
            INSERT INTO escalation_queue
                (run_id, task_id, claim_id, payload, confidence)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                run_id,
                task_id,
                claim_id,
                json.dumps(payload),
                confidence,
            ),
        ).fetchone()
        return row[0]


def list_pending_escalations(dsn: str) -> list[EscalationRecord]:
    with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        rows = conn.execute(
            "SELECT * FROM escalation_queue WHERE status = 'pending' ORDER BY created_at"
        ).fetchall()
    return [EscalationRecord(**row) for row in rows]


def resolve_escalation(dsn: str, escalation_id: int) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE escalation_queue SET status = 'resolved', resolved_at = now() WHERE id = %s",
            (escalation_id,),
        )
