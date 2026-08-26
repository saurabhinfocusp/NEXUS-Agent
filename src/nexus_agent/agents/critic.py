"""Critic/XAI Agent (Constitution Art. III §1, Art. IV §3).

"Reviews outputs from Vision and Analyst for internal consistency and
confidence, triggers error-recovery re-routing when a step fails or is
implausible." XAI artifact generation (Grad-CAM++, SHAP) is Phase 4; here
it's the pass/veto/escalate control logic, applied to the confidence of the
message under review.

These thresholds are new Phase 1 engineering values -- Article XII does not
name them, so tuning them later is not a Constitutional amendment.
"""

from __future__ import annotations

from nexus_agent.agents.common import build_envelope
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, Verdict
from nexus_agent.shared.versioning import stamp

VETO_CONFIDENCE_THRESHOLD = 0.4
ESCALATE_CONFIDENCE_THRESHOLD = 0.15


def _default_verdict(confidence: float) -> Verdict:
    if confidence < ESCALATE_CONFIDENCE_THRESHOLD:
        return Verdict.ESCALATE
    if confidence < VETO_CONFIDENCE_THRESHOLD:
        return Verdict.VETO
    return Verdict.PASS


def critic_node(state: RunState) -> dict:
    reviewed = state["history"][-1]  # the agent output under review
    verdict = state.get("force_verdict") or _default_verdict(reviewed.confidence)

    to_agent = {
        Verdict.PASS: AgentName.REPORT,
        Verdict.VETO: reviewed.from_agent,
        Verdict.ESCALATE: AgentName.HUMAN_REVIEW,
    }[verdict]

    payload = {
        "component_version": stamp(AgentName.CRITIC).model_dump(mode="json"),
        "verdict": verdict.value,
        "reviewed_confidence": reviewed.confidence,
    }
    if verdict == Verdict.VETO:
        payload["objection"] = (
            f"confidence {reviewed.confidence:.2f} below veto threshold {VETO_CONFIDENCE_THRESHOLD}"
            if not state.get("force_verdict")
            else "stub: forced veto for testing the re-route path (Art. IV §3)"
        )

    envelope = build_envelope(
        state,
        from_agent=AgentName.CRITIC,
        to_agent=to_agent,
        payload=payload,
        confidence=0.9,
    )
    # Clear the test hook after one use so a forced veto/escalate doesn't loop forever.
    return {"history": [envelope], "verdict": verdict, "force_verdict": None}
