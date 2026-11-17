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

import logging
from dataclasses import dataclass
from functools import lru_cache

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

logger = logging.getLogger(__name__)


def _fixed_seed_fusion_model():
    from nexus_agent.analyst.fusion import GraphFusionTransformer
    from nexus_agent.vision.embedding import EMBEDDING_DIM

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(_FUSION_BASE_SEED)
        return GraphFusionTransformer(morphology_dim=EMBEDDING_DIM, expression_pca_dim=50)
    finally:
        torch.random.set_rng_state(rng_state)


@lru_cache(maxsize=2)
def _load_pretrained_base(uri: str):
    from nexus_agent.analyst.pretrain import load_fusion_base
    from nexus_agent.data.object_store import get_bytes

    return load_fusion_base(get_bytes(uri))


@dataclass(frozen=True)
class ActiveFusion:
    model: torch.nn.Module
    projector: object | None  # analyst/pretrain.py::ExpressionProjector
    base_id: str
    space_id: str


def active_fusion_base(morphology_model_version: str) -> ActiveFusion:
    """The fusion base this deployment runs, for embeddings from the Vision
    encoder `morphology_model_version`: the pretrained base at
    `settings.fusion_checkpoint_uri` when one is configured, else the
    fixed-seed untrained base (no projector -- `fuse()` falls back to its
    per-sample PCA). Analyst's predictions, Critic's SHAP explanations and
    the fine-tune loop must all go through here; `space_id`
    (`analyst/pretrain.py::fusion_space_id`) is how they check they agree.

    Raises `ValueError` if the pretrained base was trained on a different
    encoder's embeddings -- a deployment misconfiguration, not something to
    carry on through.
    """
    from nexus_agent.analyst.pretrain import FIXED_SEED_BASE_ID, fusion_space_id
    from nexus_agent.shared.config import settings

    if settings.fusion_checkpoint_uri:
        base = _load_pretrained_base(settings.fusion_checkpoint_uri)
        if base.morphology_model_version not in (None, morphology_model_version):
            raise ValueError(
                f"the fusion base at {settings.fusion_checkpoint_uri} was trained on "
                f"{base.morphology_model_version} embeddings, but Vision produced "
                f"{morphology_model_version}; set VISION_EMBEDDER to match the base or train a base "
                "for this encoder"
            )
        return ActiveFusion(
            base.model, base.projector, base.base_id, fusion_space_id(base.base_id, morphology_model_version)
        )
    return ActiveFusion(
        _fixed_seed_fusion_model(),
        None,
        FIXED_SEED_BASE_ID,
        fusion_space_id(FIXED_SEED_BASE_ID, morphology_model_version),
    )


class AnalystClaim(BaseModel):
    cell_id: str
    cell_type: str | None = None
    spatial_domain: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    provisional: bool
    # How well this cell's image picks out its own expression profile among
    # the sample's cells (analyst/agreement.py); 0.5 = chance. Evidence
    # reported beside the claim, not part of `confidence`. None unless
    # Vision used OmiCLIP.
    modality_agreement: float | None = Field(default=None, ge=0.0, le=1.0)


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
        if len(vision_cells) != adata.n_obs:
            # `fuse()` (analyst/fusion.py) requires vision_cells and adata.obs to be
            # 1:1 row-aligned and already raises a clear ValueError for this -- but
            # only once it's reached. Assigning a mismatched-length array straight
            # into `adata.obsm["spatial"]` fails first, inside AnnData's own
            # dimension check, with a message that doesn't explain *why* the counts
            # differ or what to do about it. Two real modalities segmented/measured
            # independently (CellPose's cell count vs. an expression matrix's
            # spot/cell count) essentially never match by coincidence -- this isn't
            # a corrupt upload, it's missing cross-modal registration (Art. XII §1).
            raise ValueError(
                f"expression data at {expression_uri!r} has no obsm['spatial'] and its "
                f"{adata.n_obs} cell(s)/spot(s) don't match Vision's {len(vision_cells)} "
                "segmented cell(s), so per-cell spatial coordinates can't be inferred "
                "from Vision's centroids. Real multimodal fusion (Art. V §1) needs "
                "either a spatially-resolved expression file that already carries its "
                "own obsm['spatial'] (Visium/MERFISH/Xenium), or an image/expression "
                "pair with a matching cell count and real cross-modal registration "
                "(Art. XII §1) -- this pipeline does not yet perform that registration "
                "itself."
            )
        # Test/demo convenience: derive spatial coords from Vision's centroids
        # when the expression data doesn't already carry its own (e.g. a
        # platform like Visium/MERFISH would already have obsm["spatial"]) --
        # only valid when the two modalities already share a cell count.
        adata.obsm["spatial"] = np.array([cell["centroid_xy"] for cell in vision_cells])

    spatial_knn_graph(adata)

    from nexus_agent.analyst.fusion import fuse

    morphology_version = vision_cells[0]["embedding_model_version"] if vision_cells else "none"
    active = active_fusion_base(morphology_version)
    fusion_model, projector, fusion_space_id = active.model, active.projector, active.space_id
    base_reasoning = [f"fusion base: {active.base_id}; morphology encoder: {morphology_version}"]
    expression_features = None
    if projector is not None:
        coverage = projector.gene_coverage(adata.var_names)
        expression_features = projector.transform(adata)
        base_reasoning.append(
            f"{coverage:.0%} of the pretrained base's {len(projector.gene_names)} genes are present in this sample; "
            "missing genes are imputed at their training mean"
        )
    else:
        base_reasoning.append("untrained fixed-seed weights: the fused representation carries no learned biology")
    fused_records = fuse(vision_cells, adata, model=fusion_model, expression_features=expression_features)

    # Phase 5 (Art. XII §6): predict cell_type from the latest *promoted*
    # fine-tune checkpoint, if one exists yet. Falls back to None (today's
    # Phase 3 placeholder behavior) before the first correction-driven
    # fine-tune cycle has been promoted -- there is nothing to predict with.
    from nexus_agent.analyst.domains import assign_domains
    from nexus_agent.learning.celltyping import (
        checkpoint_fusion_space_id,
        latest_promoted_checkpoint_bytes,
        predict_cell_types,
    )
    from nexus_agent.shared.config import settings as _settings

    reasoning = [
        f"fused {len(fused_records)} cell(s) from Vision + expression data ({expression_uri}) (Art. V §1: not provisional)",
        *base_reasoning,
    ]

    fused_matrix = np.array([r.fused_embedding for r in fused_records], dtype=np.float64)
    domains, domain_confidence, domain_reasoning = assign_domains(fused_matrix)
    reasoning.extend(domain_reasoning)

    checkpoint_bytes = latest_promoted_checkpoint_bytes(_settings.postgres_dsn)
    if checkpoint_bytes is not None and checkpoint_fusion_space_id(checkpoint_bytes) != fusion_space_id:
        reasoning.append(
            f"the promoted cell-type checkpoint was trained in fusion space "
            f"{checkpoint_fusion_space_id(checkpoint_bytes)}, not the active {fusion_space_id}; it is not "
            "applied, so no cell type is called until a checkpoint is fine-tuned and promoted in this space"
        )
        checkpoint_bytes = None
    cell_types = predict_cell_types(fused_matrix, checkpoint_bytes)
    agreement = _modality_agreement(vision_cells, adata, morphology_version, reasoning)
    if checkpoint_bytes is not None:
        reasoning.append(
            "cell_type predicted from the latest promoted Phase 5 fine-tune checkpoint (Art. XII §6); "
            "its confidence is the classifier's softmax probability for the chosen label (uncalibrated)"
        )

    # A claim asserts everything it carries, so its confidence is that of
    # its weakest part: the domain assignment and, when a cell type was
    # called, that call too.
    claims = [
        AnalystClaim(
            cell_id=r.cell_id,
            cell_type=cell_type,
            spatial_domain=domain,
            confidence=float(min(d_conf, type_conf) if type_conf is not None else d_conf),
            provisional=False,
            modality_agreement=None if agreement is None or np.isnan(agreement[i]) else float(agreement[i]),
        )
        for i, (r, domain, d_conf, (cell_type, type_conf)) in enumerate(
            zip(fused_records, domains, domain_confidence, cell_types)
        )
    ]
    confidence = float(np.mean([c.confidence for c in claims]))
    reasoning.append(
        f"per-claim confidence = min(domain, cell-type) confidence; mean {confidence:.2f}, "
        f"min {min(c.confidence for c in claims):.2f} across {len(claims)} claim(s)"
    )
    provenance_records = [
        {
            "claim_id": r.cell_id,
            "source_image_region": r.source_image_region,
            # The base id travels with the embedding: the fine-tune loop
            # may only train on embeddings from the base it will be used with.
            "source_expression_profile": {**r.source_expression_profile, "fusion_space_id": fusion_space_id},
            "fused_embedding": r.fused_embedding,
        }
        for r in fused_records
    ]
    return claims, confidence, reasoning, provenance_records


def _modality_agreement(vision_cells: list[dict], adata, morphology_version: str, reasoning: list[str]):
    """Per-cell image-expression agreement (`analyst/agreement.py`) when
    Vision embedded with OmiCLIP, else None. Best-effort: a failure leaves
    the claims without it and says so in `reasoning`.
    """
    from nexus_agent.vision.omiclip import is_omiclip_version

    if not is_omiclip_version(morphology_version):
        return None
    try:
        from nexus_agent.analyst.agreement import retrieval_agreement
        from nexus_agent.vision.omiclip import gene_sentences, get_omiclip_embedder, image_latent_from_embedding

        expression = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        text_latents = get_omiclip_embedder().text_latents(gene_sentences(adata.var_names, expression))
        image_latents = np.stack([image_latent_from_embedding(c["embedding_vector"]) for c in vision_cells])
        agreement = retrieval_agreement(image_latents, text_latents)
    except Exception as exc:  # noqa: BLE001
        logger.warning("image-expression agreement failed", exc_info=True)
        reasoning.append(f"image-expression agreement not computed ({exc})")
        return None
    if np.isnan(agreement).all():
        reasoning.append("image-expression agreement needs at least 2 cells; not computed")
        return agreement
    reasoning.append(
        f"OmiCLIP image-expression agreement (0.5 = chance): mean {np.nanmean(agreement):.2f}, "
        f"{np.nanmean(agreement < 0.5):.0%} of cells below chance; OmiCLIP was trained on Visium spots, "
        "so for single-cell crops this is unvalidated evidence, not part of confidence"
    )
    return agreement


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
