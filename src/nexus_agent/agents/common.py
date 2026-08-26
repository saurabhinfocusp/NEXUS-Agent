"""Shared helpers for agent node implementations."""

from __future__ import annotations

from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, MessageEnvelope, Verdict


def build_envelope(
    state: RunState,
    *,
    from_agent: AgentName,
    to_agent: AgentName,
    payload: dict,
    confidence: float,
) -> MessageEnvelope:
    return MessageEnvelope(
        run_id=state["run_id"],
        task_id=state["task_id"],
        from_agent=from_agent,
        to_agent=to_agent,
        payload=payload,
        confidence=confidence,
        trace_id=state["trace_id"],
    )


def is_retry_after_veto(state: RunState, agent: AgentName) -> bool:
    """True if the immediately preceding message is Critic vetoing `agent`'s
    prior output (Art. IV §3) -- i.e. this invocation is a re-route retry,
    not a first attempt.
    """
    history = state.get("history") or []
    if not history:
        return False
    last = history[-1]
    return (
        last.from_agent == AgentName.CRITIC
        and last.to_agent == agent
        and last.payload.get("verdict") == Verdict.VETO.value
    )


def resolve_stub_confidence(
    state: RunState,
    agent: AgentName,
    *,
    default: float,
    retry_confidence: float,
) -> tuple[float, list[str]]:
    """Confidence + reasoning for a stub specialist node's output.

    Retrying after a Critic veto takes priority over the test-only override
    (Art. IV §3: the agent acts on the objection rather than repeating the
    same low-confidence output) -- the override only affects the *first*
    attempt.
    """
    if is_retry_after_veto(state, agent):
        return retry_confidence, [
            "retrying after Critic's veto objection; treating it as a cue to report higher confidence"
        ]

    override = (state.get("stub_confidence_override") or {}).get(agent)
    if override is not None:
        return override, [f"confidence overridden to {override} for testing (stub_confidence_override)"]

    return default, ["stub: no real model logic yet"]
