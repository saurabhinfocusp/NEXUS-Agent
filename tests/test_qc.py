"""Unit-level tests for the QC agent (Art. III §4) -- calls `qc_node`
directly with a hand-built state, mirroring `test_critic.py`'s style, plus
its stub/real branches following `test_graph_real_pipeline.py`'s
monkeypatch-the-object-store convention for the real path so these stay
Docker-free.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from nexus_agent.agents.qc import (
    QC_FAIL_CONFIDENCE_THRESHOLD,
    QC_FLAG_CONFIDENCE_THRESHOLD,
    QCReport,
    qc_node,
)
from nexus_agent.shared.schemas import AgentName


def _base_state(**overrides):
    run_id, task_id, trace_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    state = dict(
        run_id=run_id,
        task_id=task_id,
        trace_id=trace_id,
        history=[],
        subtask_plan=[AgentName.QC, AgentName.VISION, AgentName.ANALYST],
        verdict=None,
        force_verdict=None,
        stub_confidence_override=None,
        image_uri=None,
        expression_uri=None,
    )
    state.update(overrides)
    return state


def test_stub_path_defaults_to_pass():
    state = _base_state()
    result = qc_node(state)

    assert result["qc_verdict"] == "pass"
    envelope = result["history"][0]
    assert envelope.from_agent == AgentName.QC
    assert envelope.to_agent == AgentName.VISION  # Vision is first in the plan
    report = QCReport.model_validate(envelope.payload["qc_report"])
    assert report.qc_verdict == "pass"
    assert 0.0 <= envelope.confidence <= 1.0


def test_stub_path_routes_to_analyst_when_vision_not_in_plan():
    state = _base_state(subtask_plan=[AgentName.QC, AgentName.ANALYST])
    result = qc_node(state)

    assert result["history"][0].to_agent == AgentName.ANALYST


@pytest.mark.parametrize(
    ("override", "expected_verdict"),
    [
        (0.95, "pass"),
        ((QC_FLAG_CONFIDENCE_THRESHOLD + QC_FAIL_CONFIDENCE_THRESHOLD) / 2, "flag"),
        (QC_FAIL_CONFIDENCE_THRESHOLD - 0.05, "fail"),
    ],
)
def test_stub_confidence_override_drives_verdict_bucket(override, expected_verdict):
    state = _base_state(stub_confidence_override={AgentName.QC: override})
    result = qc_node(state)

    assert result["qc_verdict"] == expected_verdict
    assert result["history"][0].payload["qc_report"]["qc_verdict"] == expected_verdict


def test_qc_never_targets_critic_as_to_agent_when_no_plan_agent_left():
    # Degenerate plan (QC only) still produces a valid, contract-satisfying
    # envelope rather than erroring -- falls back to Critic as the only
    # remaining agent name, though this shouldn't occur via plan_subtasks().
    state = _base_state(subtask_plan=[AgentName.QC])
    result = qc_node(state)
    assert result["history"][0].to_agent == AgentName.CRITIC


def _synthetic_adata(n_cells: int = 6, n_genes: int = 10) -> AnnData:
    rng = np.random.default_rng(0)
    var_names = [f"GENE{i}" for i in range(n_genes - 1)] + ["MT-CO1"]
    return AnnData(
        X=rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names),
    )


def test_real_expression_qc_computes_metrics_and_uploads_plot(monkeypatch):
    adata = _synthetic_adata()

    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr(
        "nexus_agent.data.object_store.put_bytes", lambda key, data: f"s3://fake-bucket/{key}"
    )

    state = _base_state(expression_uri="s3://fake-bucket/expr.h5ad", subtask_plan=[AgentName.QC, AgentName.ANALYST])
    result = qc_node(state)

    envelope = result["history"][0]
    report = QCReport.model_validate(envelope.payload["qc_report"])
    assert report.total_counts_mean is not None
    assert report.n_genes_by_counts_mean is not None
    assert report.pct_counts_mt_mean is not None
    assert report.qc_plot_uri == "s3://fake-bucket/qc/" + str(state["run_id"]) + "/" + str(state["task_id"]) + "/qc_plot.png"
    assert result["qc_verdict"] in ("pass", "flag", "fail")


def test_real_expression_qc_flags_high_mitochondrial_fraction(monkeypatch):
    # All-mitochondrial-gene expression profile -> pct_counts_mt close to
    # 100 -> quality score collapses toward 0 -> a "fail" verdict.
    n_cells, n_genes = 6, 4
    rng = np.random.default_rng(0)
    var_names = [f"MT-GENE{i}" for i in range(n_genes)]
    adata = AnnData(
        X=rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names),
    )

    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", lambda uri: adata)
    monkeypatch.setattr("nexus_agent.data.object_store.put_bytes", lambda key, data: "s3://fake/plot.png")

    state = _base_state(expression_uri="s3://fake-bucket/expr.h5ad", subtask_plan=[AgentName.QC, AgentName.ANALYST])
    result = qc_node(state)

    assert result["qc_verdict"] == "fail"


def test_real_image_qc_computes_focus_and_coverage_metrics(monkeypatch):
    blank_image = np.zeros((32, 32), dtype=np.uint8)  # no focus, no tissue
    monkeypatch.setattr("nexus_agent.data.object_store.get_array", lambda uri: blank_image)

    state = _base_state(image_uri="s3://fake-bucket/image.npy", subtask_plan=[AgentName.QC, AgentName.VISION])
    result = qc_node(state)

    envelope = result["history"][0]
    report = QCReport.model_validate(envelope.payload["qc_report"])
    assert report.image_focus_score is not None
    assert report.image_tissue_coverage_fraction is not None
    assert result["qc_verdict"] == "fail"  # a blank image has zero focus signal
