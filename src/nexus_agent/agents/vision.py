"""Vision Specialist Agent (Constitution Art. III §1, Art. XII §2).

Owns all image-derived evidence: segmentation, morphological feature
extraction, and image encoding. The real model stack (CellPose/StarDist,
DINOv2/VGG16) is Phase 3 work; this stub emits one synthetic
VisionCellRecord so the output *contract* is real ahead of the model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp

EMBEDDING_DIM = 1024


class VisionCellRecord(BaseModel):
    """Per-cell Vision output contract (Art. XII §2)."""

    cell_id: str
    centroid_xy: tuple[float, float]
    mask_polygon: list[tuple[float, float]]
    embedding_vector: list[float] = Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)
    embedding_model_version: str


def _stub_cell_record() -> VisionCellRecord:
    """One synthetic cell -- stands in for real segmentation/encoding output
    (Phase 3) so downstream nodes and tests have something contract-valid
    to consume.
    """
    return VisionCellRecord(
        cell_id="stub-cell-0",
        centroid_xy=(0.0, 0.0),
        mask_polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        embedding_vector=[0.0] * EMBEDDING_DIM,
        embedding_model_version="stub-0.1.0",
    )


def vision_node(state: RunState) -> dict:
    to_agent = AgentName.ANALYST if AgentName.ANALYST in state["subtask_plan"] else AgentName.CRITIC
    cells = [_stub_cell_record()]

    envelope = build_envelope(
        state,
        from_agent=AgentName.VISION,
        to_agent=to_agent,
        payload={
            "component_version": stamp(AgentName.VISION).model_dump(mode="json"),
            "cells": [cell.model_dump(mode="json") for cell in cells],
        },
        confidence=0.5,
    )
    return {"history": [envelope]}
