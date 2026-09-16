"""Fast, Docker-free graph tests using MemorySaver.

Confirms the pipeline (Coordinator -> Vision -> Analyst -> Critic) flows
end to end, that Coordinator's subtask plan correctly skips agents whose
modality wasn't requested, and that Critic's veto path re-routes to the
originating agent with the objection attached, rather than terminating
(Art. IV §3).
"""

import uuid

import pytest

from nexus_agent.graph.build import compile_with_memory
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal, Verdict


def _initial_state(**overrides):
    state = dict(
        run_id=uuid.uuid4(),
        task_id=uuid.uuid4(),
        trace_id=uuid.uuid4(),
        goal=AnalyticalGoal(sample_id="sample-1", modalities=["image", "expression"]),
        history=[],
        verdict=None,
        force_verdict=None,
    )
    state.update(overrides)
    return state


def _config(run_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


def test_happy_path_flows_through_all_four_agents_to_report():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(_initial_state(run_id=run_id), config=_config(run_id))

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.VISION,
        AgentName.ANALYST,
        AgentName.CRITIC,
    ]
    assert result["verdict"] == Verdict.PASS
    assert result["history"][-1].to_agent == AgentName.REPORT
    for message in result["history"]:
        assert 0.0 <= message.confidence <= 1.0


@pytest.mark.parametrize(
    ("modalities", "expected_agents"),
    [
        (["image", "expression"], [AgentName.COORDINATOR, AgentName.VISION, AgentName.ANALYST, AgentName.CRITIC]),
        (["image"], [AgentName.COORDINATOR, AgentName.VISION, AgentName.CRITIC]),
        (["expression"], [AgentName.COORDINATOR, AgentName.ANALYST, AgentName.CRITIC]),
    ],
)
def test_coordinator_subtask_plan_skips_agents_with_no_requested_modality(modalities, expected_agents):
    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=modalities)

    result = graph.invoke(_initial_state(run_id=run_id, goal=goal), config=_config(run_id))

    assert [m.from_agent for m in result["history"]] == expected_agents
    assert result["verdict"] == Verdict.PASS


def test_forced_veto_reroutes_to_originating_agent_then_completes():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(
        _initial_state(run_id=run_id, force_verdict=Verdict.VETO),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    # coordinator, vision, analyst, critic(veto), analyst(re-run), critic(pass)
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.VISION,
        AgentName.ANALYST,
        AgentName.CRITIC,
        AgentName.ANALYST,
        AgentName.CRITIC,
    ]

    first_critic_message = result["history"][3]
    assert first_critic_message.to_agent == AgentName.ANALYST
    assert "objection" in first_critic_message.payload

    assert result["verdict"] == Verdict.PASS
    assert result["history"][-1].to_agent == AgentName.REPORT


def test_forced_escalate_routes_to_human_review():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(
        _initial_state(run_id=run_id, force_verdict=Verdict.ESCALATE),
        config=_config(run_id),
    )

    assert result["verdict"] == Verdict.ESCALATE
    assert result["history"][-1].to_agent == AgentName.HUMAN_REVIEW


def test_no_agent_bypasses_critic_to_reach_report():
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(_initial_state(run_id=run_id), config=_config(run_id))

    non_critic_messages = result["history"][:-1]
    assert all(m.to_agent != AgentName.REPORT for m in non_critic_messages)


def test_stub_path_never_adds_qc_to_the_plan():
    # QC is only prepended when a real data URI is present (mirrors every
    # other real-path gate) -- the Phase 0/1 synthetic-stub path (no
    # image_uri/expression_uri, what this whole file otherwise exercises)
    # must stay exactly as it was before QC existed.
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(_initial_state(run_id=run_id), config=_config(run_id))

    assert AgentName.QC not in [m.from_agent for m in result["history"]]


def test_qc_pass_routes_to_analyst_then_flows_through_unchanged(monkeypatch):
    # Expression-only modality keeps this test on Analyst's lightweight
    # ingest-only real path (no Vision -> no CellPose/VGG16 model weights
    # needed), so this stays fast/Docker-free like the rest of this file.
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(50, size=(4, 40)).astype(np.float32),  # high enough total_counts to pass QC
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
    )
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr("nexus_agent.data.object_store.put_bytes", lambda key, data: "s3://fake/qc_plot.png")

    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=["expression"])

    result = graph.invoke(
        _initial_state(run_id=run_id, goal=goal, expression_uri="s3://fake-bucket/expr.h5ad"),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [AgentName.COORDINATOR, AgentName.QC, AgentName.ANALYST, AgentName.CRITIC]
    assert result["qc_verdict"] == "pass"


def test_qc_fail_routes_directly_to_human_review_without_vision_or_critic(monkeypatch):
    import numpy as np

    blank_image = np.zeros((16, 16), dtype=np.uint8)  # zero focus signal -> QC fail
    monkeypatch.setattr("nexus_agent.data.object_store.get_array", lambda uri: blank_image)

    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=["image"])

    result = graph.invoke(
        _initial_state(run_id=run_id, goal=goal, image_uri="s3://fake-bucket/image.npy"),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [AgentName.COORDINATOR, AgentName.QC]
    assert result["qc_verdict"] == "fail"
    assert result["history"][-1].to_agent == AgentName.HUMAN_REVIEW
    assert not any(m.from_agent in (AgentName.VISION, AgentName.ANALYST, AgentName.CRITIC) for m in result["history"])


def test_run_spatial_analysis_flag_routes_analyst_through_spatial_before_critic(monkeypatch):
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(50, size=(6, 20)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(6)]),
    )
    adata.obsm["spatial"] = rng.random((6, 2)) * 100
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr("nexus_agent.data.object_store.put_bytes", lambda key, data: "s3://fake/qc_plot.png")

    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=["expression"], run_spatial_analysis=True)

    result = graph.invoke(
        _initial_state(run_id=run_id, goal=goal, expression_uri="s3://fake-bucket/expr.h5ad"),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.QC,
        AgentName.ANALYST,
        AgentName.SPATIAL,
        AgentName.CRITIC,
    ]
    assert result["verdict"] == Verdict.PASS


def test_run_spatial_analysis_flag_off_skips_spatial():
    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=["expression"], run_spatial_analysis=False)

    result = graph.invoke(_initial_state(run_id=run_id, goal=goal), config=_config(run_id))

    assert AgentName.SPATIAL not in [m.from_agent for m in result["history"]]


def _patch_biology_network_backends(monkeypatch):
    # Fast/Docker-free: no real Enrichr/g:Profiler/STRING/literature-RAG
    # network calls -- every backend is short-circuited to "nothing found",
    # which is still a valid (stub-fallback) Biology output.
    import gprofiler
    import gseapy
    import requests

    monkeypatch.setattr(gseapy, "enrichr", lambda **kwargs: type("R", (), {"results": __import__("pandas").DataFrame()})())

    class _FakeGProfiler:
        def __init__(self, *args, **kwargs):
            pass

        def profile(self, organism, query):
            return __import__("pandas").DataFrame()

    monkeypatch.setattr(gprofiler, "GProfiler", _FakeGProfiler)

    def _fake_post(url, data=None, timeout=None):
        raise RuntimeError("network disabled in fast tests")

    monkeypatch.setattr(requests, "post", _fake_post)


def test_run_enrichment_analysis_flag_routes_analyst_through_biology_before_critic(monkeypatch):
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(50, size=(6, 20)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(20)]),
    )
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr("nexus_agent.data.object_store.put_bytes", lambda key, data: "s3://fake/qc_plot.png")
    _patch_biology_network_backends(monkeypatch)

    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=["expression"], run_enrichment_analysis=True)

    result = graph.invoke(
        _initial_state(run_id=run_id, goal=goal, expression_uri="s3://fake-bucket/expr.h5ad"),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.QC,
        AgentName.ANALYST,
        AgentName.BIOLOGY,
        AgentName.CRITIC,
    ]
    # All enrichment backends are mocked to return nothing here, so Biology
    # falls back to its low-confidence placeholder claim (agents/biology.py)
    # -- below Critic's escalate threshold by design, so this terminates via
    # human_review rather than looping (see that fallback's docstring).
    assert result["verdict"] == Verdict.ESCALATE


def test_run_enrichment_analysis_flag_off_skips_biology():
    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(sample_id="sample-1", modalities=["expression"], run_enrichment_analysis=False)

    result = graph.invoke(_initial_state(run_id=run_id, goal=goal), config=_config(run_id))

    assert AgentName.BIOLOGY not in [m.from_agent for m in result["history"]]


def test_both_spatial_and_enrichment_flags_chain_analyst_spatial_biology_critic(monkeypatch):
    import numpy as np
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(50, size=(6, 20)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"GENE{i}" for i in range(20)]),
    )
    adata.obsm["spatial"] = rng.random((6, 2)) * 100
    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr("nexus_agent.data.object_store.put_bytes", lambda key, data: "s3://fake/qc_plot.png")
    _patch_biology_network_backends(monkeypatch)

    graph = compile_with_memory()
    run_id = uuid.uuid4()
    goal = AnalyticalGoal(
        sample_id="sample-1",
        modalities=["expression"],
        run_spatial_analysis=True,
        run_enrichment_analysis=True,
    )

    result = graph.invoke(
        _initial_state(run_id=run_id, goal=goal, expression_uri="s3://fake-bucket/expr.h5ad"),
        config=_config(run_id),
    )

    from_agents = [m.from_agent for m in result["history"]]
    assert from_agents == [
        AgentName.COORDINATOR,
        AgentName.QC,
        AgentName.ANALYST,
        AgentName.SPATIAL,
        AgentName.BIOLOGY,
        AgentName.CRITIC,
    ]
