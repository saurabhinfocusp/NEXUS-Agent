"""Fast, Docker-free graph tests using MemorySaver.

Confirms the stub pipeline (Coordinator -> Vision -> Analyst -> Critic)
flows end to end and that Critic's veto path re-routes to the originating
agent with the objection attached, rather than terminating (Art. IV §3).
"""

import uuid

from nexus_agent.graph.build import compile_with_memory
from nexus_agent.shared.schemas import AgentName, Verdict


def _initial_state(**overrides):
    state = dict(
        run_id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        trace_id=uuid.uuid4(),
        history=[],
        verdict=None,
        force_verdict=None,
    )
    state.update(overrides)
    return state


def _config(run_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


def test_happy_path_flows_through_all_four_agents_to_report():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(_initial_state(run_id=run_id), config=_config(run_id))

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.VISION,
        AgentName.ANALYST,
        AgentName.CRITIC,
    ]
    assert result["verdict"] == Verdict.PASS
    assert result["history"][-1].to_agent == AgentName.REPORT
    for message in result["history"]:
        assert 0.0 <= message.confidence <= 1.0


def test_forced_veto_reroutes_to_originating_agent_then_completes():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(
        _initial_state(run_id=run_id, force_verdict=Verdict.VETO),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    # coordinator, vision, analyst, critic(veto), analyst(re-run), critic(pass)
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.VISION,
        AgentName.ANALYST,
        AgentName.CRITIC,
        AgentName.ANALYST,
        AgentName.CRITIC,
    ]

    first_critic_message = result["history"][3]
    assert first_critic_message.to_agent == AgentName.ANALYST
    assert "objection" in first_critic_message.payload

    assert result["verdict"] == Verdict.PASS
    assert result["history"][-1].to_agent == AgentName.REPORT


def test_forced_escalate_routes_to_human_review():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(
        _initial_state(run_id=run_id, force_verdict=Verdict.ESCALATE),
        config=_config(run_id),
    )

    assert result["verdict"] == Verdict.ESCALATE
    assert result["history"][-1].to_agent == AgentName.HUMAN_REVIEW


def test_no_agent_bypasses_critic_to_reach_report():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(_initial_state(run_id=run_id), config=_config(run_id))

    non_critic_messages = result["history"][:-1]
    assert all(m.to_agent != AgentName.REPORT for m in non_critic_messages)
