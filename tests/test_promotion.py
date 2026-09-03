"""`evaluate_promotion_gate` is a fast pure-function test (Constitution
Art. XII §6's promotion gate); `promote_checkpoint`'s DB write is an
integration test requiring `docker compose up -d` (Postgres reachable).
"""

from __future__ import annotations

import json

import psycopg
import pytest

from nexus_agent.learning.promotion import evaluate_promotion_gate, promote_checkpoint
from nexus_agent.shared.config import settings


def test_promotion_gate_passes_when_new_scores_match_or_exceed_prior():
    # Scores are percentage points (e.g. macro-F1 * 100), matching the
    # Art. XII §6 default `max_regression=1.0` ("no more than 1 percentage
    # point of absolute regression tolerated").
    result = evaluate_promotion_gate(
        new_scores={"H&E": 82.0, "MERFISH": 75.0},
        prior_scores={"H&E": 80.0, "MERFISH": 75.0},
    )
    assert result.passed is True
    assert result.per_platform["H&E"]["within_tolerance"] is True
    assert result.per_platform["MERFISH"]["delta"] == 0.0


def test_promotion_gate_passes_within_one_point_regression_tolerance():
    result = evaluate_promotion_gate(
        new_scores={"H&E": 79.5},  # 0.5pt regression, within the 1.0pt default tolerance
        prior_scores={"H&E": 80.0},
    )
    assert result.passed is True
    assert result.per_platform["H&E"]["within_tolerance"] is True
    assert result.per_platform["H&E"]["delta"] == pytest.approx(-0.5)


def test_promotion_gate_fails_on_regression_exceeding_tolerance():
    # Scores are percentage points (e.g. macro-F1 * 100), matching the
    # Art. XII §6 default `max_regression=1.0` ("no more than 1 percentage
    # point"): a 5-point regression must fail against a 1.0pt tolerance.
    result = evaluate_promotion_gate(
        new_scores={"H&E": 75.0},
        prior_scores={"H&E": 80.0},
        max_regression=1.0,
    )
    assert result.passed is False
    assert result.per_platform["H&E"]["within_tolerance"] is False
    assert result.per_platform["H&E"]["delta"] == pytest.approx(-5.0)


def test_promotion_gate_fails_when_a_validated_platform_is_missing_from_new_scores():
    result = evaluate_promotion_gate(
        new_scores={"H&E": 0.85},  # IMC was previously validated but isn't measured here
        prior_scores={"H&E": 0.80, "IMC": 0.70},
    )
    assert result.passed is False
    assert result.per_platform["IMC"]["new"] is None
    assert result.per_platform["IMC"]["delta"] is None
    assert result.per_platform["IMC"]["within_tolerance"] is False
    # The platform that WAS measured and passes shouldn't be dragged down.
    assert result.per_platform["H&E"]["within_tolerance"] is True


def test_promotion_gate_notes_new_platform_coverage_without_penalizing_it():
    result = evaluate_promotion_gate(
        new_scores={"H&E": 0.85, "Visium": 0.60},
        prior_scores={"H&E": 0.80},
    )
    assert result.passed is True
    assert result.per_platform["Visium"]["prior"] is None
    assert result.per_platform["Visium"]["within_tolerance"] is True


@pytest.mark.integration
class TestPromoteCheckpointDB:
    @pytest.fixture(autouse=True)
    def _require_postgres(self):
        try:
            with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
                pass
        except psycopg.OperationalError as exc:
            pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")

    def test_promote_checkpoint_writes_promoted_and_benchmark_scores(self):
        with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
            row = conn.execute(
                """
                INSERT INTO finetune_runs (n_corrections, trigger_reason, checkpoint_uri, label_map, promoted)
                VALUES (%s, %s, %s, %s, false)
                RETURNING id
                """,
                (5, "test setup", "s3://nexus-agent/checkpoints/test/checkpoint.pt", json.dumps({"0": "T-cell"})),
            ).fetchone()
            finetune_run_id = row[0]

        result = promote_checkpoint(
            settings.postgres_dsn,
            finetune_run_id,
            new_scores={"H&E": 0.85},
            prior_scores={"H&E": 0.80},
        )
        assert result.passed is True

        with psycopg.connect(settings.postgres_dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
            db_row = conn.execute(
                "SELECT promoted, benchmark_scores FROM finetune_runs WHERE id = %s", (finetune_run_id,)
            ).fetchone()

        assert db_row["promoted"] is True
        assert db_row["benchmark_scores"]["new_scores"] == {"H&E": 0.85}
        assert db_row["benchmark_scores"]["prior_scores"] == {"H&E": 0.80}
