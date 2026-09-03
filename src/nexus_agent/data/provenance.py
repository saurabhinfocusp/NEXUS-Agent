"""Provenance log writer (Constitution Art. V §2, Art. XII §8).

"Every fused representation must retain a pointer back to its source image
region and source expression profile, so that any downstream explanation
can be traced to both original modalities." Writes into the `provenance_log`
table `db/schema.sql` already creates; called from `analyst_node`
(`agents/analyst.py`) only when a real fused claim is produced.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import psycopg


def record_provenance(
    dsn: str,
    *,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    claim_id: str,
    source_image_region: dict[str, Any] | None,
    source_expression_profile: dict[str, Any] | None,
    component: str,
    component_version: str,
    fused_embedding: list[float] | None = None,
) -> None:
    """`fused_embedding` (Art. XII §6) is the 128-dim fused representation at
    claim time, so Phase 5's LoRA fine-tune loop can recover
    `(cell_id -> fused_embedding)` training pairs from a later correction
    without a second object-store round trip. Nullable/optional -- only
    real (non-stub) fused claims populate it.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO provenance_log
                (run_id, task_id, claim_id, source_image_region,
                 source_expression_profile, component, component_version,
                 fused_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                task_id,
                claim_id,
                json.dumps(source_image_region) if source_image_region is not None else None,
                json.dumps(source_expression_profile) if source_expression_profile is not None else None,
                component,
                component_version,
                json.dumps(fused_embedding) if fused_embedding is not None else None,
            ),
        )


def fetch_provenance(dsn: str, run_id: uuid.UUID) -> list[dict[str, Any]]:
    with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        return conn.execute(
            "SELECT * FROM provenance_log WHERE run_id = %s ORDER BY id", (run_id,)
        ).fetchall()
