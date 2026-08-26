"""Integration test: requires `docker compose up -d` (Postgres reachable).

Confirms Phase 0's stated exit criterion: a run persisted via PostgresSaver
has its full history queryable from the checkpointer afterward, not only
the final result (Art. IV §1).
"""

import uuid

import psycopg
import pytest

from nexus_agent.agents.critic import VETO_CONFIDENCE_THRESHOLD
from nexus_agent.graph.build import compile_with_postgres
from nexus_agent.shared.config import settings
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal, Verdict

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")


def _initial_state(run_id: uuid.UUID, **overrides) -> dict:
    state = dict(
        run_id=run_id,
        task_id=uuid.uuid4(),
        trace_id=uuid.uuid4(),
        goal=AnalyticalGoal(sample_id="sample-1", modalities=["image", "expression"]),
        history=[],
        verdict=None,
        force_verdict=None,
        stub_confidence_override=None,
    )
    state.update(overrides)
    return state


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


def test_run_history_is_reconstructable_at_a_point_strictly_between_veto_and_retry():
    """Art. IV §1: the checkpointer must let you inspect state at ANY
    intermediate point, not just before/after the full run -- specifically
    including mid-way through a veto/re-route cycle (Art. IV §3).
    """
    run_id = uuid.uuid4()
    config = {"configurable": {"thread_id": str(run_id)}}
    low_confidence = VETO_CONFIDENCE_THRESHOLD - 0.1

    with compile_with_postgres(settings.postgres_dsn) as graph:
        result = graph.invoke(
            _initial_state(run_id, stub_confidence_override={AgentName.ANALYST: low_confidence}),
            config=config,
        )
        assert result["verdict"] == Verdict.PASS

        history_snapshots = list(graph.get_state_history(config))

    # newest-first; find the snapshot taken right after Critic's veto message
    # (4 messages: coordinator, vision, analyst[low-confidence], critic[veto])
    # -- strictly before the retry (analyst again) and final pass exist.
    mid_run_snapshot = next(s for s in history_snapshots if len(s.values.get("history", [])) == 4)

    mid_history = mid_run_snapshot.values["history"]
    assert [m.from_agent for m in mid_history] == [
        AgentName.COORDINATOR,
        AgentName.VISION,
        AgentName.ANALYST,
        AgentName.CRITIC,
    ]
    assert mid_history[-1].payload["verdict"] == Verdict.VETO.value
    assert mid_run_snapshot.values["verdict"] == Verdict.VETO
