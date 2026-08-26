"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Phase 0's stated exit criterion: a run persisted via PostgresSaver
has its full history queryable from the checkpointer afterward, not only
the final result (Art. IV §1).
"""

import uuid

import psycopg
import pytest

from nexus_agent.graph.build import compile_with_postgres
from nexus_agent.shared.config import settings
from nexus_agent.shared.schemas import AgentName, Verdict

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


def _initial_state(run_id: uuid.UUID) -> dict:
    return dict(
        run_id=run_id,
        task_id=uuid.uuid4(),
        trace_id=uuid.uuid4(),
        history=[],
        verdict=None,
        force_verdict=None,
    )


def test_run_history_is_queryable_from_postgres_checkpointer_after_completion():
    run_id = uuid.uuid4()
    config = {"configurable": {"thread_id": str(run_id)}}

    with compile_with_postgres(settings.postgres_dsn) as graph:
        result = graph.invoke(_initial_state(run_id), config=config)
        assert result["verdict"] == Verdict.PASS

        history_snapshots = list(graph.get_state_history(config))

    # get_state_history yields newest-first; the graph took 5 steps
    # (coordinator, vision, analyst, critic, report) from the initial state.
    assert len(history_snapshots) >= 5

    earliest_history_len = len(history_snapshots[-1].values.get("history", []))
    latest_history_len = len(history_snapshots[0].values.get("history", []))
    assert earliest_history_len == 0
    assert latest_history_len == 4  # coordinator, vision, analyst, critic messages

    final_message = history_snapshots[0].values["history"][-1]
    assert final_message.from_agent == AgentName.CRITIC
    assert final_message.to_agent == AgentName.REPORT
