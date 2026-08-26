"""Analyst Agent (Constitution Art. III §1, Art. V §1).

Owns spatial-omics reasoning: ingestion, statistical analysis, cell-type
and spatial-domain assignment on the fused representation, and biological
interpretation. Real ingestion/fusion/typing (Scanpy/Squidpy/xSiGra) is
Phase 3 work; this stub emits one synthetic AnalystClaim, contract-first.

Per Art. V §1, a claim computed without Vision's evidence (single-modality)
is provisional -- `provisional` reflects whether Vision ran for this task,
not whether real fusion happened (no fusion is implemented yet either way).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp


class AnalystClaim(BaseModel):
    cell_id: str
    cell_type: str | None = None
    spatial_domain: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    provisional: bool


def _stub_claim(*, provisional: bool) -> AnalystClaim:
    return AnalystClaim(
        cell_id="stub-cell-0",
        cell_type=None,
        spatial_domain=None,
        confidence=0.5,
        provisional=provisional,
    )


def analyst_node(state: RunState) -> dict:
    vision_ran = AgentName.VISION in state["subtask_plan"]
    claims = [_stub_claim(provisional=not vision_ran)]
    confidence, reasoning = resolve_stub_confidence(
        state, AgentName.ANALYST, default=0.5, retry_confidence=0.6
    )

    envelope = build_envelope(
        state,
        from_agent=AgentName.ANALYST,
        to_agent=AgentName.CRITIC,
        payload={
            "component_version": stamp(AgentName.ANALYST).model_dump(mode="json"),
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "reasoning": reasoning,
        },
        confidence=confidence,
    )
    return {"history": [envelope]}
