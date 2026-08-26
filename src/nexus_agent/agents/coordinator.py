"""Coordinator Agent (Constitution Art. III §1).

"Receives the user's analytical goal, decomposes it into subtasks, routes
work to the appropriate specialist agent. Holds no domain-specific model
weights itself; its authority is procedural, not scientific." Decomposition
here is a lookup from requested modalities to the ordered list of
specialist agents that must run -- no model inference.
"""

from __future__ import annotations

from nexus_agent.agents.common import build_envelope
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal
from nexus_agent.shared.versioning import stamp


def plan_subtasks(goal: AnalyticalGoal) -> list[AgentName]:
    """Which specialist agents must run, in order, for the given modalities.

    Vision owns image-derived evidence, Analyst owns expression ingestion
    and (when both modalities are present) fusion -- so Vision, when it
    runs, always precedes Analyst.
    """
    modalities = set(goal.modalities)
    plan: list[AgentName] = []
    if "image" in modalities:
        plan.append(AgentName.VISION)
    if "expression" in modalities:
        plan.append(AgentName.ANALYST)
    return plan


def coordinator_node(state: RunState) -> dict:
    goal = state["goal"]
    subtask_plan = plan_subtasks(goal)

    # Retained structured reasoning (Art. IV §2) for a non-trivial decision.
    reasoning = [
        f"requested modalities: {', '.join(goal.modalities)}",
        f"decomposed to subtask plan: {', '.join(agent.value for agent in subtask_plan)}",
    ]

    envelope = build_envelope(
        state,
        from_agent=AgentName.COORDINATOR,
        to_agent=subtask_plan[0],
        payload={
            "component_version": stamp(AgentName.COORDINATOR).model_dump(mode="json"),
            "subtask_plan": [agent.value for agent in subtask_plan],
            "reasoning": reasoning,
        },
        confidence=1.0,
    )
    return {"history": [envelope], "subtask_plan": subtask_plan}
