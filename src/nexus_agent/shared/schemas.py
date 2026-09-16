"""Inter-agent message envelope (Constitution Art. XII §4).

Every message that crosses the bus (src/nexus_agent/bus/client.py) or flows
through the LangGraph state (src/nexus_agent/graph/) must be a valid
MessageEnvelope. `confidence` is required and bounded so no agent can emit a
decision without a confidence/uncertainty field (Art. III §3).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentName(StrEnum):
    COORDINATOR = "coordinator"
    QC = "qc"
    VISION = "vision"
    ANALYST = "analyst"
    SPATIAL = "spatial"
    BIOLOGY = "biology"
    CRITIC = "critic"
    REPORT = "report"
    HUMAN_REVIEW = "human_review"


class Verdict(StrEnum):
    """Critic's routing decision (Art. XII §4)."""

    PASS = "pass"
    VETO = "veto"
    ESCALATE = "escalate"


class MessageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    task_id: uuid.UUID
    from_agent: AgentName
    to_agent: AgentName
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    trace_id: uuid.UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


Modality = Literal["image", "expression"]


class AnalyticalGoal(BaseModel):
    """The Coordinator's input (Art. III §1): what to analyze and with which
    modalities. Lives here (not in agents/coordinator.py) because RunState
    (graph/state.py) needs it as a real import, and agents/coordinator.py
    already imports RunState -- keeping it in shared/schemas.py avoids a
    coordinator<->state import cycle.
    """

    sample_id: str
    modalities: list[Modality] = Field(min_length=1)
    # Opt-in, off by default so existing goals/tests are unaffected. Both
    # require AgentName.ANALYST already in the plan (expression data is
    # their input) -- plan_subtasks() no-ops rather than erroring otherwise.
    run_spatial_analysis: bool = False
    run_enrichment_analysis: bool = False
