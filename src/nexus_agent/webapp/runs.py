"""Async pipeline-run tracking for the upload-and-view frontend.

`graph.invoke()` on the real pipeline (CellPose + VGG16 inference) takes
minutes on CPU, so `webapp/api.py`'s `POST /api/runs` cannot block on it --
it creates a `pipeline_runs` row (`db/schema.sql`) and returns immediately,
`webapp/pipeline_runner.py` updates that row as the background job
progresses, and the frontend polls `GET /api/runs/{run_id}` against it.
Mirrors `data/provenance.py`'s exact psycopg connect/execute/autocommit +
JSONB pattern.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

RunStatus = Literal["pending", "running", "done", "failed"]


class PipelineRun(BaseModel):
    run_id: uuid.UUID
    task_id: uuid.UUID
    sample_id: str
    image_uri: str
    expression_uri: str
    status: RunStatus
    verdict: str | None = None
    claims: list[dict[str, Any]] | None = None
    report_html: str | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


def create_run(
    dsn: str,
    *,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    sample_id: str,
    image_uri: str,
    expression_uri: str,
) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs
                (run_id, task_id, sample_id, image_uri, expression_uri, status)
            VALUES (%s, %s, %s, %s, %s, 'pending')
            """,
            (run_id, task_id, sample_id, image_uri, expression_uri),
        )


def mark_running(dsn: str, run_id: uuid.UUID) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("UPDATE pipeline_runs SET status = 'running' WHERE run_id = %s", (run_id,))


def mark_done(
    dsn: str,
    run_id: uuid.UUID,
    *,
    verdict: str,
    claims: list[dict[str, Any]],
    report_html: str | None,
) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            """
            UPDATE pipeline_runs
            SET status = 'done', verdict = %s, claims = %s, report_html = %s,
                completed_at = now()
            WHERE run_id = %s
            """,
            (verdict, json.dumps(claims), report_html, run_id),
        )


def mark_failed(dsn: str, run_id: uuid.UUID, error: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            """
            UPDATE pipeline_runs
            SET status = 'failed', error = %s, completed_at = now()
            WHERE run_id = %s
            """,
            (error, run_id),
        )


def fetch_run(dsn: str, run_id: uuid.UUID) -> PipelineRun | None:
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id = %s", (run_id,)).fetchone()
    return PipelineRun.model_validate(row) if row is not None else None
