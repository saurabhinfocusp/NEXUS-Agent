"""Analyst Agent (Constitution Art. III §1, Art. V §1).

Owns spatial-omics reasoning: ingestion, statistical analysis, cell-type
and spatial-domain assignment on the fused representation, and biological
interpretation. When `state["expression_uri"]` is set, this runs the real
Phase 3 ingestion (`analyst/ingestion.py`, Scanpy/Squidpy) and, if Vision
also ran, real fusion (`analyst/fusion.py`) -- otherwise it falls back to
the Phase 0/1 single-synthetic-claim stub, so fast/unit tests stay Docker-
and model-free.

Per Art. V §1, a claim computed without Vision's evidence (single-modality)
is provisional. Real cell typing/domain assignment itself remains a
placeholder either way -- Article XII names a model stack for segmentation,
embedding, and fusion, but not for typing (see ROADMAP.md's Phase 3 scope
notes); what's real now is that claims are built from an actual fused
representation rather than a hardcoded confidence.
"""

from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, MessageEnvelope
from nexus_agent.shared.versioning import stamp

# Phase 5 (Art. XII §6) needs the SAME cell's fused embedding to be
# reproducible across separate pipeline runs -- otherwise a correction
# trained against one run's fused_embedding can never generalize to a later
# run's prediction for "the same" cell, since `fuse()`'s default
# `GraphFusionTransformer()` is freshly random-initialized on every call
# (Phase 3 disclosed this: "weights are randomly initialized... proves the
# architecture is implemented correctly, not that fusion quality has been
# validated"). Seeding the base deterministically doesn't change that
# disclosure -- it's still an untrained baseline -- it just makes the
# baseline a fixed (if arbitrary) function of its input instead of a new
# random one each call, which is what Phase 5's fine-tune loop needs to be
# coherent at all. A real *trained* fusion checkpoint (not just this fixed
# seed) remains future work, same as Phase 3 disclosed.
_FUSION_BASE_SEED = 20260827


def _fixed_seed_fusion_model():
    from nexus_agent.analyst.fusion import GraphFusionTransformer
    from nexus_agent.vision.embedding import EMBEDDING_DIM

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(_FUSION_BASE_SEED)
        return GraphFusionTransformer(morphology_dim=EMBEDDING_DIM, expression_pca_dim=50)
    finally:
        torch.random.set_rng_state(rng_state)


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


def _last_vision_message(state: RunState) -> MessageEnvelope:
    for message in reversed(state["history"]):
        if message.from_agent == AgentName.VISION:
            return message
    raise LookupError("expected a Vision message in history but found none")


def _ingest_and_fuse(state: RunState, expression_uri: str, vision_ran: bool):
    from nexus_agent.analyst.ingestion import normalize, spatial_knn_graph
    from nexus_agent.data.object_store import get_anndata

    adata = get_anndata(expression_uri)
    normalize(adata)

    if not vision_ran:
        claims = [
            AnalystClaim(cell_id=str(name), confidence=0.5, provisional=True) for name in adata.obs_names
        ]
        reasoning = [f"ingested expression-only data ({expression_uri}); no Vision evidence -> provisional (Art. V §1)"]
        return claims, 0.5, reasoning, []

    vision_cells = _last_vision_message(state).payload["cells"]  # list[dict], VisionCellRecord.model_dump()

    if "spatial" not in adata.obsm:
        # Test/demo convenience: derive spatial coords from Vision's centroids
        # when the expression data doesn't already carry its own (e.g. a
        # platform like Visium/MERFISH would already have obsm["spatial"]).
        adata.obsm["spatial"] = np.array([cell["centroid_xy"] for cell in vision_cells])

    spatial_knn_graph(adata)

    from nexus_agent.analyst.fusion import fuse

    fused_records = fuse(vision_cells, adata, model=_fixed_seed_fusion_model())

    # Phase 5 (Art. XII §6): predict cell_type from the latest *promoted*
    # fine-tune checkpoint, if one exists yet. Falls back to None (today's
    # Phase 3 placeholder behavior) before the first correction-driven
    # fine-tune cycle has been promoted -- there is nothing to predict with.
    from nexus_agent.learning.celltyping import latest_promoted_checkpoint_bytes, predict_cell_type
    from nexus_agent.shared.config import settings as _settings

    checkpoint_bytes = latest_promoted_checkpoint_bytes(_settings.postgres_dsn)
    claims = [
        AnalystClaim(
            cell_id=r.cell_id,
            cell_type=predict_cell_type(r.fused_embedding, checkpoint_bytes),
            confidence=0.7,
            provisional=False,
        )
        for r in fused_records
    ]
    reasoning = [
        f"fused {len(fused_records)} cell(s) from Vision + expression data ({expression_uri}) (Art. V §1: not provisional)"
    ]
    if checkpoint_bytes is not None:
        reasoning.append("cell_type predicted from the latest promoted Phase 5 fine-tune checkpoint (Art. XII §6)")
    provenance_records = [
        {
            "claim_id": r.cell_id,
            "source_image_region": r.source_image_region,
            "source_expression_profile": r.source_expression_profile,
            "fused_embedding": r.fused_embedding,
        }
        for r in fused_records
    ]
    return claims, 0.7, reasoning, provenance_records


def _record_provenance(state: RunState, provenance_records: list[dict]) -> None:
    from nexus_agent.data.provenance import record_provenance
    from nexus_agent.shared.config import settings
    from nexus_agent.shared.versioning import stamp as _stamp

    version = _stamp(AgentName.ANALYST).version
    for record in provenance_records:
        record_provenance(
            settings.postgres_dsn,
            run_id=state["run_id"],
            task_id=state["task_id"],
            claim_id=record["claim_id"],
            source_image_region=record["source_image_region"],
            source_expression_profile=record["source_expression_profile"],
            component=AgentName.ANALYST.value,
            component_version=version,
            fused_embedding=record.get("fused_embedding"),
        )


def analyst_node(state: RunState) -> dict:
    vision_ran = AgentName.VISION in state["subtask_plan"]
    expression_uri = state.get("expression_uri")

    if expression_uri:
        claims, confidence, reasoning, provenance_records = _ingest_and_fuse(state, expression_uri, vision_ran)
    else:
        claims = [_stub_claim(provisional=not vision_ran)]
        confidence, reasoning = resolve_stub_confidence(state, AgentName.ANALYST, default=0.5, retry_confidence=0.6)
        provenance_records = []

    if provenance_records:
        _record_provenance(state, provenance_records)

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
