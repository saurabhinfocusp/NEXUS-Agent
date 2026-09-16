"""Biology (Enrichment) Agent (Constitution Art. III §4, Art. V §1).

Runs after Spatial (or directly after Analyst if Spatial wasn't requested),
consuming the expression data Analyst already ingested to derive gene-set
enrichment support for the sample's biology. Two enrichment backends are
run behind the same `EnrichmentClaim` output shape, so Critic/report don't
need to know which one produced a given claim:

- `gseapy.enrichr()` against the Enrichr, KEGG_2021_Human, Reactome_2022,
  and MSigDB Hallmark_2020 gene-set libraries (one API covers all four by
  selecting a different library name).
- g:Profiler (`gprofiler-official`), added as a second backend since it was
  named explicitly and isn't redundant -- different default gene-set
  versions/species handling than gseapy/Enrichr.

STRING's public REST API is queried for protein-interaction network
support -- best-effort, wrapped in its own try/except exactly like
`agents/critic.py::_generate_evidence_bundles` already does for its own
network/model calls, since this is the one specialist agent that depends
on live network access by design. Literature support reuses
`xai/literature_rag.py::retrieve_citations` rather than duplicating RAG
logic; results are folded into `reasoning`, not a new EnrichmentClaim
field, since Critic's own evidence-bundle generation already attaches a
citation artifact per claim independently (Art. VI §3).

Like Spatial, Biology *is* a biological-claim producer (Art. III §2), so
its output routes through Critic exactly like Vision/Analyst/Spatial
claims do today.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp

# Enrichr library selection (Art. XII does not name these; new Phase-N
# engineering values disclosed like every other new threshold/list in this
# codebase). One gseapy call per library, since gseapy.enrichr() takes a
# single gene-set-library name per call.
ENRICHR_LIBRARIES = ["KEGG_2021_Human", "Reactome_2022", "MSigDB_Hallmark_2020"]
STRING_API_URL = "https://string-db.org/api/json/network"
N_TOP_GENES = 50


class EnrichmentClaim(BaseModel):
    """Biology agent's output contract. `cell_id` is deliberately reused
    (not renamed) to hold the pathway/term identifier -- Critic's
    `_generate_evidence_bundles` reads `claim["cell_id"]` generically as
    the claim's id for every claim-producing agent; an enrichment result
    has no per-cell identity, so this keeps that lookup agent-agnostic
    instead of special-casing Biology there (see agents/critic.py).
    """

    cell_id: str
    pathway: str
    gene_set: list[str]
    p_value: float = Field(ge=0.0, le=1.0)
    fdr: float | None = Field(default=None, ge=0.0, le=1.0)
    source_db: str
    confidence: float = Field(ge=0.0, le=1.0)


def _stub_claim() -> EnrichmentClaim:
    return EnrichmentClaim(
        cell_id="stub-pathway-0",
        pathway="stub-pathway-0",
        gene_set=["STUB1"],
        p_value=0.5,
        fdr=0.5,
        source_db="stub",
        confidence=0.5,
    )


def _select_gene_set(adata, n_genes: int = N_TOP_GENES) -> list[str]:
    """Highly-variable genes as the practical proxy for "the sample's genes
    of biological interest" -- there is no differential-expression/marker
    step upstream yet to draw a more targeted list from (Analyst's
    `cell_type` is itself often `None` pre-Phase-5-promotion; see
    agents/analyst.py), so this is a real, disclosed placeholder input the
    same way Analyst's own cell-typing placeholder is disclosed there.
    """
    import scanpy as sc

    n_top = min(n_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
    return [str(g) for g in adata.var_names[adata.var["highly_variable"]]]


def _confidence_from_pvalue(p_value: float) -> float:
    return max(0.0, min(1.0, 1.0 - p_value))


def _run_gseapy_enrichr(genes: list[str]) -> tuple[list[EnrichmentClaim], list[str]]:
    reasoning: list[str] = []
    claims: list[EnrichmentClaim] = []
    if not genes:
        return claims, reasoning

    import gseapy as gp

    for library in ENRICHR_LIBRARIES:
        try:
            enrichment = gp.enrichr(gene_list=genes, gene_sets=[library], organism="human", outdir=None)
            results = enrichment.results
            for _, row in results.iterrows():
                p_value = float(row["P-value"])
                fdr = float(row["Adjusted P-value"]) if "Adjusted P-value" in row else None
                genes_hit = str(row["Genes"]).split(";") if row.get("Genes") else []
                claims.append(
                    EnrichmentClaim(
                        cell_id=str(row["Term"]),
                        pathway=str(row["Term"]),
                        gene_set=genes_hit,
                        p_value=p_value,
                        fdr=fdr,
                        source_db=f"Enrichr:{library}",
                        confidence=_confidence_from_pvalue(fdr if fdr is not None else p_value),
                    )
                )
            reasoning.append(f"gseapy.enrichr against {library} returned {len(results)} term(s)")
        except Exception as exc:  # noqa: BLE001
            reasoning.append(f"gseapy.enrichr against {library} failed/unavailable ({exc}); skipped")

    return claims, reasoning


def _run_gprofiler(genes: list[str]) -> tuple[list[EnrichmentClaim], list[str]]:
    reasoning: list[str] = []
    claims: list[EnrichmentClaim] = []
    if not genes:
        return claims, reasoning

    try:
        from gprofiler import GProfiler

        client = GProfiler(return_dataframe=True)
        results = client.profile(organism="hsapiens", query=genes)
        for _, row in results.iterrows():
            p_value = float(row["p_value"])
            claims.append(
                EnrichmentClaim(
                    cell_id=str(row["native"]),
                    pathway=str(row["name"]),
                    gene_set=genes,
                    p_value=p_value,
                    fdr=p_value,  # g:Profiler's p_value is already multiple-testing corrected
                    source_db="g:Profiler",
                    confidence=_confidence_from_pvalue(p_value),
                )
            )
        reasoning.append(f"g:Profiler returned {len(results)} term(s)")
    except Exception as exc:  # noqa: BLE001
        reasoning.append(f"g:Profiler query failed/unavailable ({exc}); skipped")

    return claims, reasoning


def _string_interaction_support(genes: list[str]) -> tuple[dict | None, str]:
    """Best-effort STRING REST API call -- network-optional by design, same
    try/except convention `agents/critic.py` uses for its own external
    calls. Returns (support dict or None, one reasoning line).
    """
    if not genes:
        return None, "no gene set to query STRING with"

    import requests

    try:
        response = requests.post(
            STRING_API_URL,
            data={"identifiers": "%0d".join(genes[:20]), "species": 9606},
            timeout=5,
        )
        response.raise_for_status()
        interactions = response.json()
        support = {"interaction_count": len(interactions)}
        return support, f"STRING reported {len(interactions)} interaction(s) among the top gene set"
    except Exception as exc:  # noqa: BLE001
        return None, f"STRING REST API unavailable ({exc}); interaction-network support omitted"


def _literature_support(pathways: list[str]) -> str:
    """Reuses `xai/literature_rag.py::retrieve_citations` rather than
    duplicating RAG logic -- folded into reasoning, since Critic's own
    per-claim evidence-bundle generation already attaches a citation
    artifact independently (Art. VI §3).
    """
    if not pathways:
        return "no pathway/term to retrieve literature support for"

    try:
        from nexus_agent.shared.config import settings
        from nexus_agent.xai.literature_rag import HashingLiteratureEmbedder, retrieve_citations

        query_text = f"{pathways[0]} pathway in spatial biology"
        result = retrieve_citations(settings.postgres_dsn, query_text, HashingLiteratureEmbedder())
        if result.no_supporting_literature_retrieved:
            return f"no supporting literature retrieved for {pathways[0]!r}"
        return f"{len(result.citations)} literature citation(s) retrieved for {pathways[0]!r}"
    except Exception as exc:  # noqa: BLE001
        return f"literature retrieval unavailable ({exc})"


def _run_enrichment(expression_uri: str) -> tuple[list[EnrichmentClaim], float, list[str]]:
    from nexus_agent.analyst.ingestion import normalize
    from nexus_agent.data.object_store import get_anndata

    adata = get_anndata(expression_uri)
    normalize(adata)

    genes = _select_gene_set(adata)
    reasoning = [f"selected {len(genes)} highly-variable gene(s) from {expression_uri} as the enrichment input"]

    enrichr_claims, enrichr_reasoning = _run_gseapy_enrichr(genes)
    gprofiler_claims, gprofiler_reasoning = _run_gprofiler(genes)
    reasoning.extend(enrichr_reasoning)
    reasoning.extend(gprofiler_reasoning)

    claims = enrichr_claims + gprofiler_claims

    _string_support, string_reasoning = _string_interaction_support(genes)
    reasoning.append(string_reasoning)

    top_pathways = [c.pathway for c in sorted(claims, key=lambda c: c.p_value)[:1]]
    reasoning.append(_literature_support(top_pathways))

    if claims:
        confidence = sum(c.confidence for c in claims) / len(claims)
    else:
        claims = [_stub_claim()]
        # Below Critic's escalate threshold (agents/critic.py's
        # ESCALATE_CONFIDENCE_THRESHOLD = 0.15), not merely its veto
        # threshold -- Biology's real path recomputes this exact same
        # result deterministically on a retry, so a veto-band confidence
        # here would loop forever (Critic vetoes back to Biology, Biology
        # reports the same confidence, ad infinitum). Mirrors
        # agents/vision.py's zero-cells-found fallback (0.05), which
        # escalates to human_review instead of retrying pointlessly.
        confidence = 0.1
        reasoning.append("no enrichment results from any backend for this gene set; falling back to a placeholder claim")

    return claims, confidence, reasoning


def biology_node(state: RunState) -> dict:
    expression_uri = state.get("expression_uri")

    if expression_uri:
        claims, confidence, reasoning = _run_enrichment(expression_uri)
    else:
        claims = [_stub_claim()]
        confidence, reasoning = resolve_stub_confidence(state, AgentName.BIOLOGY, default=0.5, retry_confidence=0.6)

    envelope = build_envelope(
        state,
        from_agent=AgentName.BIOLOGY,
        to_agent=AgentName.CRITIC,
        payload={
            "component_version": stamp(AgentName.BIOLOGY).model_dump(mode="json"),
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "reasoning": reasoning,
        },
        confidence=confidence,
    )
    return {"history": [envelope]}
