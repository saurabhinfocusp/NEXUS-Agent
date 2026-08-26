"""Real (non-force_verdict) error-recovery path (Art. IV §3).

Uses `stub_confidence_override` to make a specialist agent's first attempt
genuinely low-confidence, so Critic's actual threshold logic (not the
force_verdict test hook) triggers the veto -- and confirms the re-routed
agent acts on the objection (reports higher confidence on retry) rather
than just receiving it and repeating the same output.
"""

import uuid

from nexus_agent.agents.critic import VETO_CONFIDENCE_THRESHOLD
from nexus_agent.graph.build import compile_with_memory
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal, Verdict


def _initial_state(**overrides):
    state = dict(
        run_id=uuid.uuid4(),
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


def _config(run_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


def test_genuinely_low_confidence_triggers_real_veto_and_never_reaches_report_unflagged():
    graph = compile_with_memory()
    run_id = uuid.uuid4()
    low_confidence = VETO_CONFIDENCE_THRESHOLD - 0.1

    result = graph.invoke(
        _initial_state(run_id=run_id, stub_confidence_override={AgentName.ANALYST: low_confidence}),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    # coordinator, vision, analyst(low-confidence), critic(veto, real logic),
    # analyst(retry), critic(pass)
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.VISION,
        AgentName.ANALYST,
        AgentName.CRITIC,
        AgentName.ANALYST,
        AgentName.CRITIC,
    ]

    first_analyst_message, first_critic_message = result["history"][2], result["history"][3]
    assert first_analyst_message.confidence == low_confidence
    assert first_critic_message.to_agent == AgentName.ANALYST  # vetoed back, not force_verdict-driven
    assert "objection" in first_critic_message.payload
    assert "reasoning" in first_critic_message.payload

    retry_message = result["history"][4]
    assert retry_message.confidence > low_confidence  # agent acted on the objection, not just received it

    assert result["verdict"] == Verdict.PASS
    # the low-confidence output never reached report -- critic intercepted it first
    assert first_analyst_message.to_agent != AgentName.REPORT


def test_coordinator_and_critic_payloads_carry_structured_reasoning():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(_initial_state(run_id=run_id), config=_config(run_id))

    coordinator_message = result["history"][0]
    critic_message = result["history"][-1]
    assert coordinator_message.payload["reasoning"]
    assert critic_message.payload["reasoning"]
