"""Stub Coordinator/Vision/Analyst/Critic nodes (Constitution Art. III).

No real model/domain logic yet — that lands per-agent starting Phase 1
(agent boundaries) and Phase 3 (Vision/Analyst model stack). What's real
here is the *contract*: every node emits a schema-valid MessageEnvelope
with a mandatory confidence field, and no node routes directly to a
"final" outcome except via Critic (Art. III §2).
"""

from __future__ import annotations

from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, MessageEnvelope, Verdict
from nexus_agent.shared.versioning import stamp


def _envelope(
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


def coordinator_node(state: RunState) -> dict:
    envelope = _envelope(
        state,
        from_agent=AgentName.COORDINATOR,
        to_agent=AgentName.VISION,
        payload={
            "component_version": stamp(AgentName.COORDINATOR).model_dump(mode="json"),
            "note": "stub: task decomposed, routed to vision",
        },
        confidence=1.0,
    )
    return {"history": [envelope]}


def vision_node(state: RunState) -> dict:
    envelope = _envelope(
        state,
        from_agent=AgentName.VISION,
        to_agent=AgentName.ANALYST,
        payload={
            "component_version": stamp(AgentName.VISION).model_dump(mode="json"),
            "note": "stub: no real segmentation/encoding yet (Phase 3)",
        },
        confidence=0.5,
    )
    return {"history": [envelope]}


def analyst_node(state: RunState) -> dict:
    envelope = _envelope(
        state,
        from_agent=AgentName.ANALYST,
        to_agent=AgentName.CRITIC,
        payload={
            "component_version": stamp(AgentName.ANALYST).model_dump(mode="json"),
            "note": "stub: no real fusion/typing yet (Phase 3)",
        },
        confidence=0.5,
    )
    return {"history": [envelope]}


def critic_node(state: RunState) -> dict:
    verdict = state.get("force_verdict") or Verdict.PASS
    originator = state["history"][-1].from_agent  # the agent whose output is under review

    to_agent = {
        Verdict.PASS: AgentName.REPORT,
        Verdict.VETO: originator,
        Verdict.ESCALATE: AgentName.HUMAN_REVIEW,
    }[verdict]

    payload = {
        "component_version": stamp(AgentName.CRITIC).model_dump(mode="json"),
        "verdict": verdict.value,
    }
    if verdict == Verdict.VETO:
        payload["objection"] = "stub: forced veto for testing the re-route path (Art. IV §3)"

    envelope = _envelope(
        state,
        from_agent=AgentName.CRITIC,
        to_agent=to_agent,
        payload=payload,
        confidence=0.9,
    )
    # Clear the test hook after one use so a forced veto/escalate doesn't loop forever.
    return {"history": [envelope], "verdict": verdict, "force_verdict": None}


def report_node(state: RunState) -> dict:
    """Terminal stub for the `pass` path. Real report assembly is a later phase."""
    return {}


def human_review_node(state: RunState) -> dict:
    """Terminal stub for the `escalate` path. Real review queue is Phase 5."""
    return {}
