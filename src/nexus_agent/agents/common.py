"""Shared helper for agent node implementations."""

from __future__ import annotations

from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, MessageEnvelope


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
