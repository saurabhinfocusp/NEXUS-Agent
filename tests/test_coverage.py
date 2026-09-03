"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Art. XII §7's "Explainability coverage ... measured not sampled"
metric is computed from real `xai_evidence` rows, not estimated.
"""

import json
import uuid

import psycopg
import pytest

from nexus_agent.xai.coverage import explainability_coverage
from nexus_agent.shared.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


def _insert_row(run_id: uuid.UUID, claim_id: str, artifacts_expected: list[str], artifacts_present: list[str]) -> None:
    with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO xai_evidence
                (run_id, task_id, claim_id, claim_type, confidence,
                 artifacts_expected, artifacts_present, component_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                uuid.uuid4(),
                claim_id,
                "cell_type_call",
                0.8,
                json.dumps(artifacts_expected),
                json.dumps(artifacts_present),
                "test-v1",
            ),
        )


def test_explainability_coverage_matches_hand_calculation():
    run_id = uuid.uuid4()
    expected = ["heatmap", "shap_top_genes", "citations"]

    # 2 fully covered, 1 under-covered (missing "citations") -> 2/3
    _insert_row(run_id, "cell-1", expected, ["heatmap", "shap_top_genes", "citations"])
    _insert_row(run_id, "cell-2", expected, ["heatmap", "shap_top_genes", "citations", "extra"])
    _insert_row(run_id, "cell-3", expected, ["heatmap", "shap_top_genes"])

    coverage = explainability_coverage(settings.postgres_dsn, run_id)

    assert coverage == pytest.approx(2 / 3)


def test_explainability_coverage_is_one_when_no_rows():
    assert explainability_coverage(settings.postgres_dsn, uuid.uuid4()) == 1.0
