"""Expert correction interface (Constitution Art. VII §1).

"When a human reviewer corrects an agent's claim, that correction must be
captured as a first-class, structured record -- not just a note in a
ticket -- so it can later be replayed as a training pair." Writes into the
`correction_log` table `db/schema.sql` already creates.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal

import psycopg
from pydantic import BaseModel


class CorrectionRequest(BaseModel):
    run_id: uuid.UUID
    task_id: uuid.UUID
    claim_id: str
    field: Literal["cell_type", "spatial_domain", "interpretation"]
    reviewer: str
    original_value: Any
    corrected_value: Any
    reason: str | None = None
    tissue_type: str | None = None


class CorrectionRecord(BaseModel):
    id: int
    run_id: uuid.UUID
    task_id: uuid.UUID
    claim_id: str
    field: Literal["cell_type", "spatial_domain", "interpretation"]
    reviewer: str
    original_value: Any
    corrected_value: Any
    reason: str | None = None
    tissue_type: str | None = None
    created_at: datetime


def submit_correction(dsn: str, request: CorrectionRequest) -> int:
    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            """
            INSERT INTO correction_log
                (run_id, claim_id, reviewer, original_value, corrected_value,
                 reason, task_id, field, tissue_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                request.run_id,
                request.claim_id,
                request.reviewer,
                json.dumps(request.original_value),
                json.dumps(request.corrected_value),
                request.reason,
                request.task_id,
                request.field,
                request.tissue_type,
            ),
        ).fetchone()
        return row[0]


def fetch_corrections(
    dsn: str,
    *,
    since: datetime | None = None,
    run_id: uuid.UUID | None = None,
) -> list[CorrectionRecord]:
    clauses = []
    params: list[Any] = []
    if since is not None:
        clauses.append("created_at >= %s")
        params.append(since)
    if run_id is not None:
        clauses.append("run_id = %s")
        params.append(run_id)

    query = "SELECT * FROM correction_log"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at"

    with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        rows = conn.execute(query, params).fetchall()

    return [CorrectionRecord(**row) for row in rows]
