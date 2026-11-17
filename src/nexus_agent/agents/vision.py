"""Vision Specialist Agent (Constitution Art. III §1, Art. XII §2).

Owns all image-derived evidence: segmentation, morphological feature
extraction, and image encoding. When `state["image_uri"]` is set, this runs
the real model stack (CellPose segmentation + the configured embedder --
OmiCLIP by default, VGG16 or DINOv2 -- `vision/segmentation.py`,
`vision/embedding.py::get_embedder`); otherwise it
falls back to the Phase 0/1 single-synthetic-cell stub, so fast/unit tests
stay Docker- and model-free.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp
from nexus_agent.vision.embedding import EMBEDDING_DIM


# Reported only when the segmenter exposes no probability map to measure
# from (see vision/segmentation.py::segment_cells_with_probability). New
# engineering value, not an Article XII number.
UNMEASURED_SEGMENTATION_CONFIDENCE = 0.5


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


def _segment_and_embed(image_uri: str) -> tuple[list[VisionCellRecord], dict[str, float], float, list[str]]:
    """Real Phase 3 path: fetch the image from object storage, segment with
    CellPose, embed every cell crop in one batch with the configured
    embedder (Art. XII §2 as amended in v1.3), and build real
    VisionCellRecords.
    """
    from nexus_agent.data.object_store import get_array
    from nexus_agent.vision.embedding import get_embedder
    from nexus_agent.vision.segmentation import (
        cell_confidences,
        cells_from_mask,
        crop_patch,
        segment_cells_with_probability,
    )

    image = get_array(image_uri)
    label_mask, cell_probability = segment_cells_with_probability(image)
    cell_infos = cells_from_mask(label_mask)

    if not cell_infos:
        return [], {}, 0.05, [f"CellPose found no cells in {image_uri}"]

    embedder = get_embedder()
    embeddings = embedder.embed_patches([crop_patch(image, info["bbox"]) for info in cell_infos])
    records = [
        VisionCellRecord(
            cell_id=info["cell_id"],
            centroid_xy=info["centroid_xy"],
            mask_polygon=info["mask_polygon"],
            embedding_vector=embedding.tolist(),
            embedding_model_version=embedder.model_version,
        )
        for info, embedding in zip(cell_infos, embeddings)
    ]

    reasoning = [f"CellPose segmented {len(records)} cell(s) from {image_uri}", f"embedded via {embedder.model_version}"]
    if cell_probability is None:
        per_cell: dict[str, float] = {}
        confidence = UNMEASURED_SEGMENTATION_CONFIDENCE
        reasoning.append(
            "CellPose returned no cell-probability map, so segmentation confidence was not measured; "
            f"reporting the fixed fallback {UNMEASURED_SEGMENTATION_CONFIDENCE}"
        )
    else:
        per_cell = cell_confidences(label_mask, cell_probability)
        values = np.array([per_cell[r.cell_id] for r in records])
        confidence = float(values.mean())
        reasoning.append(
            f"segmentation confidence = mean CellPose cell probability inside each mask: "
            f"mean {confidence:.2f}, min {values.min():.2f} across {len(values)} cell(s)"
        )
    return records, per_cell, confidence, reasoning


def vision_node(state: RunState) -> dict:
    to_agent = AgentName.ANALYST if AgentName.ANALYST in state["subtask_plan"] else AgentName.CRITIC
    image_uri = state.get("image_uri")

    if image_uri:
        cells, cell_confidence, confidence, reasoning = _segment_and_embed(image_uri)
    else:
        cells = [_stub_cell_record()]
        cell_confidence = {}
        confidence, reasoning = resolve_stub_confidence(state, AgentName.VISION, default=0.5, retry_confidence=0.6)

    envelope = build_envelope(
        state,
        from_agent=AgentName.VISION,
        to_agent=to_agent,
        payload={
            "component_version": stamp(AgentName.VISION).model_dump(mode="json"),
            "cells": [cell.model_dump(mode="json") for cell in cells],
            # Per-cell segmentation confidence, keyed by cell_id. Kept beside
            # `cells` rather than inside each record: the per-cell record is
            # Art. XII §2's fixed output contract.
            "cell_confidence": cell_confidence,
            "reasoning": reasoning,
        },
        confidence=confidence,
    )
    return {"history": [envelope]}
