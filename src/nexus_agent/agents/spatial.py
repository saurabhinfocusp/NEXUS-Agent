"""Spatial Transcriptomics Agent (Constitution Art. III §4, Art. V §1).

Runs after Analyst, on an opt-in basis (`AnalyticalGoal.run_spatial_analysis`),
adding real spatial-biology analysis that doesn't exist anywhere else in the
repo: niche/domain characterization via Squidpy's `nhood_enrichment` and
`co_occurrence`, and spatially variable genes via `spatial_autocorr`
(Moran's I). Reuses `analyst/ingestion.py::spatial_knn_graph` -- the exact
same k-NN graph construction Analyst's own real path already builds into
`adata.obsp["spatial_connectivities"]` -- rather than inventing a second,
possibly-divergent k-NN implementation.

Unlike QC, Spatial *is* a biological-claim producer (Art. III §2: a new
specialist may never bypass Critic with a final biological claim), so its
output routes through Critic exactly like Vision/Analyst claims do today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp

# New Phase-N engineering value (Article XII does not name a Spatial
# confidence), same disclosure convention as agents/critic.py's thresholds.
SPATIAL_REAL_CLAIM_CONFIDENCE = 0.65


class SpatialClaim(BaseModel):
    """Spatial agent's per-cell output contract: which niche/domain a cell
    was assigned to, plus the Squidpy statistic(s) supporting that
    assignment.
    """

    cell_id: str
    niche_label: str
    supporting_stat: dict
    confidence: float = Field(ge=0.0, le=1.0)


def _stub_claim() -> SpatialClaim:
    return SpatialClaim(
        cell_id="stub-cell-0",
        niche_label="stub-niche-0",
        supporting_stat={"stub": True},
        confidence=0.5,
    )


def _last_message_from(state: RunState, agent: AgentName):
    return next((m for m in reversed(state["history"]) if m.from_agent == agent), None)


def _prepare_adata(state: RunState, expression_uri: str):
    """Fetch + normalize + spatial-graph the same way Analyst's real path
    does (Art. V §1), so Spatial's niche/domain stats are computed over the
    identical graph, not a second divergent one.
    """
    from nexus_agent.analyst.ingestion import normalize, spatial_knn_graph
    from nexus_agent.data.object_store import get_anndata

    adata = get_anndata(expression_uri)
    normalize(adata)

    if "spatial" not in adata.obsm:
        vision_message = _last_message_from(state, AgentName.VISION)
        if vision_message is not None:
            cells = vision_message.payload.get("cells", [])
            if cells:
                adata.obsm["spatial"] = np.array([cell["centroid_xy"] for cell in cells])

    spatial_knn_graph(adata)
    return adata


def _assign_niches(adata) -> np.ndarray:
    """A lightweight KMeans-on-expression-PCA clustering to give Squidpy's
    `nhood_enrichment`/`co_occurrence` the categorical cluster key they
    require. Real cell-typing (Analyst's `cell_type` claim) is often `None`
    before a Phase 5 fine-tune checkpoint has been promoted (see
    agents/analyst.py), so this can't simply reuse Analyst's cell_type
    labels -- this is deliberately a cheap, real, unsupervised stand-in
    for "spatial domain," using only sklearn (already a dependency).
    """
    from sklearn.cluster import KMeans

    from nexus_agent.analyst.ingestion import expression_pca

    pcs = expression_pca(adata)
    n_clusters = max(1, min(4, adata.n_obs))
    if n_clusters == 1:
        return np.zeros(adata.n_obs, dtype=int)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(pcs)
    return labels


def _run_spatial_stats(state: RunState, expression_uri: str) -> tuple[list[SpatialClaim], float, list[str]]:
    import squidpy as sq

    reasoning: list[str] = []
    adata = _prepare_adata(state, expression_uri)

    cluster_labels = _assign_niches(adata)
    adata.obs["niche"] = pd.Categorical([f"niche_{label}" for label in cluster_labels])
    reasoning.append(
        f"assigned {len(set(cluster_labels))} spatial niche cluster(s) via KMeans-on-expression_pca "
        f"({expression_uri}) as the categorical input squidpy's cluster-key stats require"
    )

    nhood_zscore_by_niche: dict[str, float] = {}
    try:
        sq.gr.nhood_enrichment(adata, cluster_key="niche", seed=0, show_progress_bar=False)
        zscore = adata.uns["niche_nhood_enrichment"]["zscore"]
        categories = list(adata.obs["niche"].cat.categories)
        nhood_zscore_by_niche = {cat: float(zscore[i, i]) for i, cat in enumerate(categories)}
        reasoning.append("squidpy.gr.nhood_enrichment computed niche self-enrichment z-scores")
    except Exception as exc:  # noqa: BLE001
        reasoning.append(f"squidpy.gr.nhood_enrichment not applicable/failed ({exc}); niche z-score omitted")

    co_occurrence_computed = False
    try:
        sq.gr.co_occurrence(adata, cluster_key="niche")
        co_occurrence_computed = "niche_co_occurrence" in adata.uns
        reasoning.append("squidpy.gr.co_occurrence computed niche spatial co-occurrence")
    except Exception as exc:  # noqa: BLE001
        reasoning.append(f"squidpy.gr.co_occurrence not applicable/failed ({exc}); co-occurrence omitted")

    moran_top_gene: dict | None = None
    try:
        genes = list(adata.var_names)[: min(20, adata.n_vars)]
        sq.gr.spatial_autocorr(adata, mode="moran", genes=genes)
        moran_df = adata.uns["moranI"]
        if len(moran_df):
            moran_top_gene = {"gene": str(moran_df.index[0]), "moran_i": float(moran_df.iloc[0]["I"])}
        reasoning.append("squidpy.gr.spatial_autocorr (Moran's I) computed spatially variable genes")
    except Exception as exc:  # noqa: BLE001
        reasoning.append(f"squidpy.gr.spatial_autocorr not applicable/failed ({exc}); Moran's I omitted")

    claims = []
    for cell_name, niche in zip(adata.obs_names, adata.obs["niche"]):
        supporting_stat: dict = {
            "nhood_enrichment_zscore": nhood_zscore_by_niche.get(str(niche)),
            "co_occurrence_computed": co_occurrence_computed,
        }
        if moran_top_gene is not None:
            supporting_stat["moran_i_top_gene"] = moran_top_gene
        claims.append(
            SpatialClaim(
                cell_id=str(cell_name),
                niche_label=str(niche),
                supporting_stat=supporting_stat,
                confidence=SPATIAL_REAL_CLAIM_CONFIDENCE,
            )
        )

    return claims, SPATIAL_REAL_CLAIM_CONFIDENCE, reasoning


def spatial_node(state: RunState) -> dict:
    to_agent = AgentName.BIOLOGY if AgentName.BIOLOGY in state["subtask_plan"] else AgentName.CRITIC
    expression_uri = state.get("expression_uri")

    if expression_uri:
        claims, confidence, reasoning = _run_spatial_stats(state, expression_uri)
    else:
        claims = [_stub_claim()]
        confidence, reasoning = resolve_stub_confidence(state, AgentName.SPATIAL, default=0.5, retry_confidence=0.6)

    envelope = build_envelope(
        state,
        from_agent=AgentName.SPATIAL,
        to_agent=to_agent,
        payload={
            "component_version": stamp(AgentName.SPATIAL).model_dump(mode="json"),
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "reasoning": reasoning,
        },
        confidence=confidence,
    )
    return {"history": [envelope]}
