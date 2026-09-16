"""Unit-level tests for the Biology (Enrichment) agent (Art. III §4, Art. V
§1) -- calls `biology_node` directly with a hand-built state, mirroring
`test_critic.py`'s style. The real path's external calls (gseapy/Enrichr,
g:Profiler, STRING's REST API, literature RAG) are all network-mocked here
per the plan's test-plan note -- only an explicit integration test should
hit the real services.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from anndata import AnnData

from nexus_agent.agents.biology import EnrichmentClaim, biology_node
from nexus_agent.shared.schemas import AgentName


def _base_state(**overrides):
    run_id, task_id, trace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    state = dict(
        run_id=run_id,
        task_id=task_id,
        trace_id=trace_id,
        history=[],
        subtask_plan=[AgentName.ANALYST, AgentName.BIOLOGY],
        verdict=None,
        force_verdict=None,
        stub_confidence_override=None,
        image_uri=None,
        expression_uri=None,
    )
    state.update(overrides)
    return state


def test_stub_path_produces_one_synthetic_claim_and_targets_critic():
    state = _base_state()
    result = biology_node(state)

    envelope = result["history"][0]
    assert envelope.from_agent == AgentName.BIOLOGY
    assert envelope.to_agent == AgentName.CRITIC
    claims = [EnrichmentClaim.model_validate(c) for c in envelope.payload["claims"]]
    assert len(claims) == 1
    assert 0.0 <= envelope.confidence <= 1.0


def test_stub_confidence_override_is_honored():
    state = _base_state(stub_confidence_override={AgentName.BIOLOGY: 0.05})
    result = biology_node(state)
    assert result["history"][0].confidence == 0.05


def _synthetic_adata(n_cells: int = 8, n_genes: int = 20) -> AnnData:
    rng = np.random.default_rng(0)
    var_names = [f"GENE{i}" for i in range(n_genes)]
    return AnnData(
        X=rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names),
    )


def _patch_network_backends(monkeypatch, *, enrichr_terms=None, gprofiler_terms=None, string_ok=False):
    import gseapy

    class _FakeEnrichment:
        def __init__(self, results):
            self.results = results

    def fake_enrichr(gene_list, gene_sets, organism, outdir=None):
        terms = enrichr_terms or []
        return _FakeEnrichment(
            pd.DataFrame(
                {
                    "Term": [t["term"] for t in terms],
                    "P-value": [t["p"] for t in terms],
                    "Adjusted P-value": [t["fdr"] for t in terms],
                    "Genes": [";".join(t["genes"]) for t in terms],
                }
            )
        )

    monkeypatch.setattr(gseapy, "enrichr", fake_enrichr)

    import gprofiler

    class _FakeGProfiler:
        def __init__(self, *args, **kwargs):
            pass

        def profile(self, organism, query):
            terms = gprofiler_terms or []
            return pd.DataFrame(
                {
                    "native": [t["native"] for t in terms],
                    "name": [t["name"] for t in terms],
                    "p_value": [t["p"] for t in terms],
                }
            )

    monkeypatch.setattr(gprofiler, "GProfiler", _FakeGProfiler)

    import requests

    def fake_post(url, data=None, timeout=None):
        class _FakeResponse:
            def raise_for_status(self):
                if not string_ok:
                    raise RuntimeError("simulated network failure")

            def json(self):
                return [{"a": "GENE0", "b": "GENE1"}]

        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    monkeypatch.setattr(
        "nexus_agent.xai.literature_rag.retrieve_citations",
        lambda dsn, query, embedder: type(
            "R", (), {"citations": [], "no_supporting_literature_retrieved": True}
        )(),
    )


def test_real_path_falls_back_to_stub_claim_when_no_backend_returns_results(monkeypatch):
    adata = _synthetic_adata()
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    _patch_network_backends(monkeypatch)

    state = _base_state(expression_uri="s3://fake-bucket/expr.h5ad")
    result = biology_node(state)

    envelope = result["history"][0]
    claims = [EnrichmentClaim.model_validate(c) for c in envelope.payload["claims"]]
    assert len(claims) == 1
    assert claims[0].source_db == "stub"
    assert any("falling back to a placeholder claim" in line for line in envelope.payload["reasoning"])


def test_real_path_aggregates_enrichr_and_gprofiler_claims(monkeypatch):
    adata = _synthetic_adata()
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    _patch_network_backends(
        monkeypatch,
        enrichr_terms=[{"term": "Pathway A", "p": 0.01, "fdr": 0.02, "genes": ["GENE0", "GENE1"]}],
        gprofiler_terms=[{"native": "GO:0001", "name": "Pathway B", "p": 0.03}],
        string_ok=True,
    )

    state = _base_state(expression_uri="s3://fake-bucket/expr.h5ad")
    result = biology_node(state)

    envelope = result["history"][0]
    claims = [EnrichmentClaim.model_validate(c) for c in envelope.payload["claims"]]
    # The same fake Enrichr term is returned for every library in
    # ENRICHR_LIBRARIES (one gseapy.enrichr() call per library) plus one
    # g:Profiler term.
    from nexus_agent.agents.biology import ENRICHR_LIBRARIES

    assert len(claims) == len(ENRICHR_LIBRARIES) + 1
    pathways = {c.pathway for c in claims}
    assert pathways == {"Pathway A", "Pathway B"}

    enrichr_claim = next(c for c in claims if c.pathway == "Pathway A")
    assert enrichr_claim.source_db.startswith("Enrichr:")
    assert enrichr_claim.gene_set == ["GENE0", "GENE1"]
    assert enrichr_claim.fdr == 0.02

    gprofiler_claim = next(c for c in claims if c.pathway == "Pathway B")
    assert gprofiler_claim.source_db == "g:Profiler"

    assert any("STRING reported" in line for line in envelope.payload["reasoning"])
    assert 0.0 <= envelope.confidence <= 1.0


def test_real_path_reasoning_notes_string_failure_gracefully(monkeypatch):
    adata = _synthetic_adata()
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    _patch_network_backends(
        monkeypatch,
        enrichr_terms=[{"term": "Pathway A", "p": 0.01, "fdr": 0.02, "genes": ["GENE0"]}],
        string_ok=False,
    )

    state = _base_state(expression_uri="s3://fake-bucket/expr.h5ad")
    result = biology_node(state)

    reasoning = result["history"][0].payload["reasoning"]
    assert any("STRING REST API unavailable" in line for line in reasoning)
