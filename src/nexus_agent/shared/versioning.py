"""Component versioning (Constitution Art. XII §9).

Every model, prompt template, and threshold named in Article XII carries an
independent semantic version, recorded in each run's provenance entry as
{component, version}. This module is the single place that registry lives,
so a node stamps its output with `stamp(AgentName.VISION)` instead of
hand-writing the pair inline.
"""

from __future__ import annotations

from pydantic import BaseModel

from nexus_agent.shared.schemas import AgentName

# Bumped as each component's real implementation lands (Phase 1+). Phase 0
# stub nodes report "0.1.0" since they carry no real model/prompt logic yet.
COMPONENT_VERSIONS: dict[AgentName, str] = {
    AgentName.COORDINATOR: "0.1.0",
    AgentName.VISION: "0.1.0",
    AgentName.ANALYST: "0.1.0",
    AgentName.CRITIC: "0.1.0",
}


class ComponentVersion(BaseModel):
    component: AgentName
    version: str


def stamp(component: AgentName) -> ComponentVersion:
    """Return the current {component, version} pair for a registered component."""
    try:
        version = COMPONENT_VERSIONS[component]
    except KeyError as exc:
        raise ValueError(f"No registered version for component {component!r}") from exc
    return ComponentVersion(component=component, version=version)
