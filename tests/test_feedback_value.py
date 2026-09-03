"""Integration test: requires `docker compose up -d` (Postgres + MinIO reachable).

Confirms Art. VII §4's per-tissue-type feedback-value report reflects a
REAL post-fine-tuning-cycle accuracy measurement, not a placeholder --
seeds corrections across two tissue types, runs a real (tiny) fine-tune
cycle, promotes it, and checks the report's `accuracy_after` accordingly.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import psycopg
import pytest

from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.data import object_store
from nexus_agent.learning.feedback_value import feedback_value_report
from nexus_agent.learning.lora_finetune import run_finetune_cycle
from nexus_agent.learning.promotion import promote_checkpoint
from nexus_agent.shared.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_infra():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")
    try:
        object_store.ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO not reachable ({exc}); run `docker compose up -d`.")


def test_feedback_value_report_reflects_real_post_finetune_accuracy():
    dsn = settings.postgres_dsn
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    rng = np.random.default_rng(99)

    center_a = rng.normal(size=FUSED_DIM) * 2
    center_b = -center_a

    # 3 corrections in "tonsil", 3 in "spleen"; T-cell/B-cell, linearly separable.
    plan = [
        ("T-cell", "tonsil"),
        ("T-cell", "tonsil"),
        ("B-cell", "tonsil"),
        ("T-cell", "spleen"),
        ("B-cell", "spleen"),
        ("B-cell", "spleen"),
    ]

    with psycopg.connect(dsn, autocommit=True) as conn:
        for i, (label, tissue) in enumerate(plan):
            claim_id = f"fv-cell-{i}"
            center = center_a if label == "T-cell" else center_b
            embedding = (center + rng.normal(size=FUSED_DIM) * 0.1).tolist()

            conn.execute(
                """
                INSERT INTO provenance_log
                    (run_id, task_id, claim_id, source_image_region,
                     source_expression_profile, component, component_version, fused_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, task_id, claim_id, json.dumps({}), json.dumps({}), "analyst", "0.1.0", json.dumps(embedding)),
            )
            conn.execute(
                """
                INSERT INTO correction_log
                    (run_id, claim_id, reviewer, original_value, corrected_value, task_id, field, tissue_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, claim_id, "dr.jane", json.dumps("Unknown"), json.dumps(label), task_id, "cell_type", tissue),
            )

    # Before any fine-tune has ever been promoted, nothing to measure yet.
    report_before = feedback_value_report(dsn)
    assert "tonsil" in report_before and "spleen" in report_before
    for tissue_type in ("tonsil", "spleen"):
        assert report_before[tissue_type]["n_corrections"] == 3
        assert report_before[tissue_type]["accuracy_before"] == 0.0
        if report_before[tissue_type]["accuracy_after"] is not None:
            # A prior test run in a shared DB may have already promoted a
            # checkpoint; either way accuracy_before's definition must hold.
            assert report_before[tissue_type]["accuracy_before"] == 0.0

    result = run_finetune_cycle(dsn, epochs=200, lr=0.05, min_corrections=1)
    promote_checkpoint(dsn, result.finetune_run_id, new_scores={"H&E": 0.85}, prior_scores={"H&E": 0.80})

    report_after = feedback_value_report(dsn)

    for tissue_type in ("tonsil", "spleen"):
        entry = report_after[tissue_type]
        assert entry["n_corrections"] == 3
        assert entry["accuracy_before"] == 0.0
        assert entry["accuracy_after"] is not None
        assert entry["accuracy_after"] == pytest.approx(1.0)
        assert entry["delta"] == pytest.approx(1.0)
