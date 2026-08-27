"""Vision Specialist Agent (Constitution Art. III §1, Art. XII §2).

Owns all image-derived evidence: segmentation, morphological feature
extraction, and image encoding. When `state["image_uri"]` is set, this runs
the real Phase 3 model stack (CellPose segmentation + VGG16/DINOv2
embedding, `vision/segmentation.py` + `vision/embedding.py`); otherwise it
falls back to the Phase 0/1 single-synthetic-cell stub, so fast/unit tests
stay Docker- and model-free.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp
from nexus_agent.vision.embedding import EMBEDDING_DIM


class VisionCellRecord(BaseModel):
    """Per-cell Vision output contract (Art. XII §2)."""

    cell_id: str
    centroid_xy: tuple[float, float]
    mask_polygon: list[tuple[float, float]]
    embedding_vector: list[float] = Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)
    embedding_model_version: str


def _stub_cell_record() -> VisionCellRecord:
    """One synthetic cell -- used when no real image is provided, so
    downstream nodes and fast tests have something contract-valid to
    consume without paying for segmentation/embedding.
    """
    return VisionCellRecord(
        cell_id="stub-cell-0",
        centroid_xy=(0.0, 0.0),
        mask_polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        embedding_vector=[0.0] * EMBEDDING_DIM,
        embedding_model_version="stub-0.1.0",
    )


def _segment_and_embed(image_uri: str) -> tuple[list[VisionCellRecord], float, list[str]]:
    """Real Phase 3 path: fetch the image from object storage, segment with
    CellPose, embed each cell crop (VGG16 default -- see ROADMAP.md's
    Phase 3 scope notes for why), and build real VisionCellRecords.
    """
    from nexus_agent.data.object_store import get_array
    from nexus_agent.vision.embedding import Vgg16Embedder
    from nexus_agent.vision.segmentation import cells_from_mask, crop_patch, segment_cells

    image = get_array(image_uri)
    label_mask = segment_cells(image)
    cell_infos = cells_from_mask(label_mask)

    embedder = Vgg16Embedder()
    records = [
        VisionCellRecord(
            cell_id=info["cell_id"],
            centroid_xy=info["centroid_xy"],
            mask_polygon=info["mask_polygon"],
            embedding_vector=embedder.embed_patch(crop_patch(image, info["bbox"])).tolist(),
            embedding_model_version=embedder.model_version,
        )
        for info in cell_infos
    ]

    if records:
        confidence = 0.8
        reasoning = [f"CellPose segmented {len(records)} cell(s) from {image_uri}", f"embedded via {embedder.model_version}"]
    else:
        confidence = 0.05
        reasoning = [f"CellPose found no cells in {image_uri}"]

    return records, confidence, reasoning


def vision_node(state: RunState) -> dict:
    to_agent = AgentName.ANALYST if AgentName.ANALYST in state["subtask_plan"] else AgentName.CRITIC
    image_uri = state.get("image_uri")

    if image_uri:
        cells, confidence, reasoning = _segment_and_embed(image_uri)
    else:
        cells = [_stub_cell_record()]
        confidence, reasoning = resolve_stub_confidence(state, AgentName.VISION, default=0.5, retry_confidence=0.6)

    envelope = build_envelope(
        state,
        from_agent=AgentName.VISION,
        to_agent=to_agent,
        payload={
            "component_version": stamp(AgentName.VISION).model_dump(mode="json"),
            "cells": [cell.model_dump(mode="json") for cell in cells],
            "reasoning": reasoning,
        },
        confidence=confidence,
    )
    return {"history": [envelope]}
