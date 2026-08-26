"""Unit-level tests for Critic's default confidence-threshold logic
(Art. III §1, Art. IV §3) -- calls critic_node directly with a hand-built
state so each confidence band is exercised independently of graph wiring
or any particular stub agent's hardcoded confidence value.
"""

import uuid

import pytest

from nexus_agent.agents.critic import (
    ESCALATE_CONFIDENCE_THRESHOLD,
    VETO_CONFIDENCE_THRESHOLD,
    critic_node,
)
from nexus_agent.shared.schemas import AgentName, MessageEnvelope, Verdict


def _state_with_reviewed_confidence(confidence: float, from_agent: AgentName = AgentName.ANALYST, **overrides):
    run_id, task_id, trace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    reviewed = MessageEnvelope(
        run_id=run_id,
        task_id=task_id,
        from_agent=from_agent,
        to_agent=AgentName.CRITIC,
        payload={},
        confidence=confidence,
        trace_id=trace_id,
    )
    state = dict(run_id=run_id, task_id=task_id, trace_id=trace_id, history=[reviewed], verdict=None, force_verdict=None)
    state.update(overrides)
    return state


def test_high_confidence_passes():
    state = _state_with_reviewed_confidence(VETO_CONFIDENCE_THRESHOLD)  # boundary: exactly at threshold passes
    result = critic_node(state)
    assert result["verdict"] == Verdict.PASS
    assert result["history"][0].to_agent == AgentName.REPORT


def test_mid_confidence_vetoes_back_to_originator():
    state = _state_with_reviewed_confidence(ESCALATE_CONFIDENCE_THRESHOLD, from_agent=AgentName.VISION)
    result = critic_node(state)
    assert result["verdict"] == Verdict.VETO
    assert result["history"][0].to_agent == AgentName.VISION
    assert "objection" in result["history"][0].payload


def test_low_confidence_escalates():
    state = _state_with_reviewed_confidence(ESCALATE_CONFIDENCE_THRESHOLD - 0.01)
    result = critic_node(state)
    assert result["verdict"] == Verdict.ESCALATE
    assert result["history"][0].to_agent == AgentName.HUMAN_REVIEW


@pytest.mark.parametrize("forced", [Verdict.VETO, Verdict.ESCALATE, Verdict.PASS])
def test_force_verdict_overrides_confidence(forced):
    state = _state_with_reviewed_confidence(0.99, force_verdict=forced)  # would otherwise pass
    result = critic_node(state)
    assert result["verdict"] == forced
    assert result["force_verdict"] is None  # consumed
