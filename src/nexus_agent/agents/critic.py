"""Critic/XAI Agent (Constitution Art. III §1, Art. IV §3, Art. VI).

"Reviews outputs from Vision and Analyst for internal consistency and
confidence, triggers error-recovery re-routing when a step fails or is
implausible... and generates the attribution evidence (Grad-CAM++, SHAP)
attached to every claim before it reaches the report" (Art. III §1).

These thresholds are new Phase 1 engineering values -- Article XII does not
name them, so tuning them later is not a Constitutional amendment.

Phase 4 XAI artifact generation (`_generate_evidence_bundles` below) is
gated to the real pipeline (`state["expression_uri"]` set, i.e. real Analyst
claims exist) -- the Phase 0/1 stub path (no image_uri/expression_uri, used
by the fast unit tests in tests/test_critic.py and tests/test_graph_smoke.py)
is completely unaffected, exactly as before this phase.
"""

from __future__ import annotations

import numpy as np

from nexus_agent.agents.common import build_envelope
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName, Verdict
from nexus_agent.shared.versioning import stamp

VETO_CONFIDENCE_THRESHOLD = 0.4
ESCALATE_CONFIDENCE_THRESHOLD = 0.15

# Phase 4 (Art. XII §5): new engineering value, not an Article XII number.
CRITIC_XAI_VERSION = "0.1.0"


def _default_verdict(confidence: float) -> Verdict:
    if confidence < ESCALATE_CONFIDENCE_THRESHOLD:
        return Verdict.ESCALATE
    if confidence < VETO_CONFIDENCE_THRESHOLD:
        return Verdict.VETO
    return Verdict.PASS


def _claim_type_for(claim: dict) -> str:
    if claim.get("niche_label"):
        return "niche_enrichment"
    if claim.get("source_db"):
        return "pathway_enrichment"
    if claim.get("cell_type"):
        return "cell_type_call"
    if claim.get("spatial_domain"):
        return "spatial_domain_definition"
    return "biomarker_association"


def _generate_evidence_bundles(state: RunState, analyst_message) -> None:
    """Real Art. VI §1 artifact generation: a Grad-CAM++ heatmap (when real
    Vision evidence exists for the cell), a SHAP gene-importance ranking
    (when real expression data AND a Phase 5 promoted classifier both
    exist -- there is no real decision to explain before that), and a RAG
    literature citation (always attempted; Art. VI §1(3)'s "no supporting
    literature retrieved" flag is itself a valid artifact, not a failure).

    Persists one `ClaimEvidenceBundle` per claim via
    `xai/claims_evidence.py::store_evidence_bundle` so it's independently
    retrievable (Art. VI §3) before `_report_node` (graph/build.py) reads it
    back. Every artifact is generated best-effort inside its own try/except
    -- a failure leaves that artifact absent (and `artifacts_expected` vs.
    `artifacts_present` honestly reflects the gap via
    `xai/coverage.py::explainability_coverage`), it never crashes the run.
    """
    expression_uri = state.get("expression_uri")
    if not expression_uri:
        return  # stub path: no real Analyst evidence to explain at all

    from nexus_agent.shared.config import settings
    from nexus_agent.xai.claims_evidence import ClaimEvidenceBundle, store_evidence_bundle
    from nexus_agent.xai.literature_rag import HashingLiteratureEmbedder, retrieve_citations

    claims = analyst_message.payload.get("claims", [])
    if not claims:
        return

    image_uri = state.get("image_uri")
    vision_cells: list[dict] = []
    if image_uri and AgentName.VISION in state["subtask_plan"]:
        vision_message = next((m for m in state["history"] if m.from_agent == AgentName.VISION), None)
        if vision_message is not None:
            vision_cells = vision_message.payload.get("cells", [])
    vision_cells_by_id = {c["cell_id"]: c for c in vision_cells}

    adata = None
    try:
        from nexus_agent.data.object_store import get_anndata

        adata = get_anndata(expression_uri)
    except Exception:
        adata = None

    classifier_head = None
    try:
        from nexus_agent.learning.celltyping import latest_promoted_checkpoint_bytes, load_checkpoint

        checkpoint_bytes = latest_promoted_checkpoint_bytes(settings.postgres_dsn)
        if checkpoint_bytes is not None:
            import torch

            clf, refiner, _label_map = load_checkpoint(checkpoint_bytes)
            classifier_head = torch.nn.Sequential(refiner, clf) if refiner is not None else clf
    except Exception:
        classifier_head = None

    literature_embedder = HashingLiteratureEmbedder()

    for claim in claims:
        claim_id = claim["cell_id"]
        artifacts_expected: list[str] = []
        artifacts_present: list[str] = []
        heatmap_uri = None
        heatmap_array = None
        shap_top_genes = None
        citations = None
        no_literature_retrieved = False

        cell = vision_cells_by_id.get(claim_id)

        # 1. Grad-CAM++ heatmap -- only applicable when real Vision evidence
        # exists for this specific cell (Art. VI §1(1): "where imaging
        # evidence was used").
        if cell is not None:
            artifacts_expected.append("heatmap")
            try:
                from nexus_agent.data.object_store import get_array
                from nexus_agent.vision.embedding import Vgg16Embedder
                from nexus_agent.vision.segmentation import crop_patch
                from nexus_agent.xai.gradcam import grad_cam_plusplus

                image = get_array(image_uri)
                xs = [p[0] for p in cell["mask_polygon"]]
                ys = [p[1] for p in cell["mask_polygon"]]
                bbox = (int(min(ys)), int(min(xs)), int(max(ys)) + 1, int(max(xs)) + 1)
                patch = crop_patch(image, bbox)
                heatmap_array = grad_cam_plusplus(Vgg16Embedder(), patch)
                artifacts_present.append("heatmap")
            except Exception:
                heatmap_array = None  # absent, honestly reflected in coverage

        # 2. SHAP gene importance -- only applicable when real expression
        # data AND a promoted classifier both exist (Art. VI §1(2): "where
        # omics evidence was used" -- and there's a real decision to explain).
        if adata is not None and classifier_head is not None and cell is not None:
            artifacts_expected.append("shap")
            try:
                from sklearn.decomposition import PCA

                from nexus_agent.agents.analyst import _fixed_seed_fusion_model
                from nexus_agent.xai.shap_explain import shap_gene_importance

                cell_index = next(i for i, c in enumerate(vision_cells) if c["cell_id"] == claim_id)
                expression_matrix = np.asarray(adata.X, dtype=np.float64)
                target_expression = expression_matrix[cell_index]

                n_comps = max(1, min(50, adata.n_obs - 1, adata.n_vars - 1))
                pca = PCA(n_components=n_comps).fit(expression_matrix)

                def _pca_transform(x, _pca=pca, _n_comps=n_comps):
                    out = _pca.transform(x)
                    if out.shape[1] < 50:
                        pad = np.zeros((out.shape[0], 50 - out.shape[1]))
                        out = np.concatenate([out, pad], axis=1)
                    return out

                # Must be the SAME fixed-seed base `agents/analyst.py` uses
                # for real predictions -- the promoted classifier was
                # trained against that coordinate space, so explaining it
                # through a different (freshly random) fusion transformer
                # would produce meaningless SHAP values.
                fusion_model = _fixed_seed_fusion_model()
                shap_top_genes = shap_gene_importance(
                    fusion_model,
                    classifier_head,
                    np.asarray(cell["embedding_vector"], dtype=np.float64),
                    expression_matrix,
                    target_expression,
                    list(adata.var_names),
                    _pca_transform,
                )
                artifacts_present.append("shap")
            except Exception:
                shap_top_genes = None  # absent, honestly reflected in coverage

        # 3. RAG literature citation -- always attempted; the "no supporting
        # literature retrieved" flag (Art. VI §1(3)) is itself a valid,
        # present artifact, not a failure -- only a hard error (e.g.
        # Postgres unreachable) leaves this artifact genuinely absent.
        artifacts_expected.append("citation")
        try:
            claim_type = _claim_type_for(claim)
            query_text = f"{claim.get('cell_type') or claim.get('spatial_domain') or claim_type} in spatial biology"
            result = retrieve_citations(settings.postgres_dsn, query_text, literature_embedder)
            citations = [c.model_dump(mode="json") for c in result.citations] or None
            no_literature_retrieved = result.no_supporting_literature_retrieved
            artifacts_present.append("citation")
        except Exception:
            no_literature_retrieved = True  # honest fallback, but not counted as "present"

        bundle = ClaimEvidenceBundle(
            claim_id=claim_id,
            claim_type=_claim_type_for(claim),
            shap_top_genes=shap_top_genes,
            citations=citations,
            no_literature_retrieved=no_literature_retrieved,
            confidence=claim.get("confidence", 0.0),
            artifacts_expected=artifacts_expected,
            artifacts_present=artifacts_present,
            component_version=CRITIC_XAI_VERSION,
        )
        try:
            store_evidence_bundle(
                settings.postgres_dsn,
                state["run_id"],
                state["task_id"],
                bundle,
                heatmap_array=heatmap_array,
            )
        except Exception:
            pass  # best-effort persistence; a failed store here doesn't block the run


def critic_node(state: RunState) -> dict:
    reviewed = state["history"][-1]  # the agent output under review
    forced = state.get("force_verdict")
    verdict = forced or _default_verdict(reviewed.confidence)

    to_agent = {
        Verdict.PASS: AgentName.REPORT,
        Verdict.VETO: reviewed.from_agent,
        Verdict.ESCALATE: AgentName.HUMAN_REVIEW,
    }[verdict]

    # Retained structured reasoning (Art. IV §2) -- persisted as part of the
    # checkpointed MessageEnvelope, not just logged for debugging.
    if forced:
        reasoning = [f"verdict forced to {verdict.value} for testing (force_verdict override)"]
    else:
        reasoning = [
            f"reviewed confidence {reviewed.confidence:.2f} vs. thresholds "
            f"(escalate < {ESCALATE_CONFIDENCE_THRESHOLD}, veto < {VETO_CONFIDENCE_THRESHOLD}) -> {verdict.value}"
        ]

    payload = {
        "component_version": stamp(AgentName.CRITIC).model_dump(mode="json"),
        "verdict": verdict.value,
        "reviewed_confidence": reviewed.confidence,
        "reasoning": reasoning,
    }
    if verdict == Verdict.VETO:
        payload["objection"] = (
            "stub: forced veto for testing the re-route path (Art. IV §3)"
            if forced
            else f"confidence {reviewed.confidence:.2f} below veto threshold {VETO_CONFIDENCE_THRESHOLD}"
        )

    if verdict == Verdict.PASS and reviewed.from_agent in {AgentName.ANALYST, AgentName.SPATIAL, AgentName.BIOLOGY}:
        _generate_evidence_bundles(state, reviewed)

    envelope = build_envelope(
        state,
        from_agent=AgentName.CRITIC,
        to_agent=to_agent,
        payload=payload,
        confidence=0.9,
    )
    # Clear the test hook after one use so a forced veto/escalate doesn't loop forever.
    return {"history": [envelope], "verdict": verdict, "force_verdict": None}
