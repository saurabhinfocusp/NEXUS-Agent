"""Regression tests for control-flow/evidence bugs found in review.

Each section pins one bug that the rest of the fast suite didn't catch:

1. QC's `fail` gate reaches `human_review` without Critic, so the escalated
   message is `history[-1]`, not `history[-2]`.
2. A QC-rejected run has no Critic verdict; the webapp must say so rather
   than report "unknown".
3. Spatial/Biology claims must reach the report, and a vetoed first attempt's
   claims must not.
4. Critic must attach evidence to every claim-producing agent's claims, not
   only to the message it happened to review last.
5. Critic's veto loop is capped.
6. Best-effort persistence failures are logged, not silently swallowed.

All fast and Docker-free: Postgres/S3 touchpoints are monkeypatched.
"""

import logging
import uuid
from types import SimpleNamespace

import pytest

from nexus_agent.agents.common import evidence_claim_id
from nexus_agent.agents.critic import MAX_VETOES_PER_AGENT, VETO_CONFIDENCE_THRESHOLD, critic_node
from nexus_agent.graph import build as graph_build
from nexus_agent.graph.build import (
    _human_review_node,
    _with_message_logging,
    collect_report_claims,
    compile_with_memory,
    escalated_message,
)
from nexus_agent.report.build import ReportClaimInput, render_report, render_report_html
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal, MessageEnvelope, Verdict
from nexus_agent.webapp.pipeline_runner import QC_FAIL_VERDICT, summarize_result

RUN_ID, TASK_ID, TRACE_ID = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


def _msg(from_agent, to_agent, *, confidence=0.7, **payload) -> MessageEnvelope:
    return MessageEnvelope(
        run_id=RUN_ID,
        task_id=TASK_ID,
        from_agent=from_agent,
        to_agent=to_agent,
        payload=payload,
        confidence=confidence,
        trace_id=TRACE_ID,
    )


def _state(history, **overrides) -> dict:
    state = dict(run_id=RUN_ID, task_id=TASK_ID, trace_id=TRACE_ID, history=history, verdict=None, force_verdict=None)
    state.update(overrides)
    return state


def _veto(agent: AgentName) -> MessageEnvelope:
    return _msg(AgentName.CRITIC, agent, confidence=0.9, verdict=Verdict.VETO.value)


ANALYST_CLAIM = {"cell_id": "cell-0", "cell_type": "T cell", "spatial_domain": None, "confidence": 0.7}
SPATIAL_CLAIM = {"cell_id": "cell-0", "niche_label": "niche-2", "supporting_stat": {}, "confidence": 0.65}
BIOLOGY_CLAIM = {
    "cell_id": "GO:0006955",
    "pathway": "immune response",
    "gene_set": ["CD3E"],
    "p_value": 0.001,
    "fdr": 0.01,
    "source_db": "GO_Biological_Process",
    "confidence": 0.9,
}


def _full_history() -> list[MessageEnvelope]:
    return [
        _msg(AgentName.COORDINATOR, AgentName.ANALYST, confidence=1.0),
        _msg(AgentName.ANALYST, AgentName.SPATIAL, claims=[ANALYST_CLAIM]),
        _msg(AgentName.SPATIAL, AgentName.BIOLOGY, confidence=0.65, claims=[SPATIAL_CLAIM]),
        _msg(AgentName.BIOLOGY, AgentName.CRITIC, confidence=0.9, claims=[BIOLOGY_CLAIM]),
    ]


# ---------------------------------------------------------------------------
# 1. human_review records the right message on both routes into it.
# ---------------------------------------------------------------------------


def test_escalated_message_is_qc_report_when_qc_fail_bypasses_critic():
    coordinator = _msg(AgentName.COORDINATOR, AgentName.QC, confidence=1.0)
    qc = _msg(AgentName.QC, AgentName.HUMAN_REVIEW, confidence=0.1, qc_report={"qc_verdict": "fail"})

    assert escalated_message(_state([coordinator, qc])) is qc


def test_escalated_message_is_the_reviewed_output_when_critic_escalates():
    analyst = _msg(AgentName.ANALYST, AgentName.CRITIC, confidence=0.05, claims=[ANALYST_CLAIM])
    critic = _msg(AgentName.CRITIC, AgentName.HUMAN_REVIEW, confidence=0.9, verdict=Verdict.ESCALATE.value)

    assert escalated_message(_state([analyst, critic])) is analyst


def test_human_review_node_persists_qc_report_not_coordinator_plan(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        "nexus_agent.review.escalation.record_escalation", lambda dsn, **kwargs: recorded.update(kwargs)
    )
    coordinator = _msg(AgentName.COORDINATOR, AgentName.QC, confidence=1.0, subtask_plan=["qc", "vision"])
    qc = _msg(AgentName.QC, AgentName.HUMAN_REVIEW, confidence=0.1, qc_report={"qc_verdict": "fail"})

    _human_review_node(_state([coordinator, qc], image_uri="s3://fake/image.npy", qc_verdict="fail"))

    assert recorded["payload"] == {"qc_report": {"qc_verdict": "fail"}}
    assert recorded["confidence"] == 0.1


# ---------------------------------------------------------------------------
# 2. The webapp's run summary.
# ---------------------------------------------------------------------------


def test_summarize_result_reports_qc_fail_instead_of_unknown():
    qc = _msg(AgentName.QC, AgentName.HUMAN_REVIEW, confidence=0.1, qc_report={"qc_verdict": "fail"})

    verdict, claims = summarize_result({"history": [qc], "verdict": None, "qc_verdict": "fail"})

    assert verdict == QC_FAIL_VERDICT
    assert claims == []


def test_summarize_result_uses_critic_verdict_and_latest_analyst_claims():
    vetoed_claim = {**ANALYST_CLAIM, "cell_type": "vetoed-guess"}
    history = [
        _msg(AgentName.ANALYST, AgentName.CRITIC, confidence=0.3, claims=[vetoed_claim]),
        _veto(AgentName.ANALYST),
        _msg(AgentName.ANALYST, AgentName.CRITIC, claims=[ANALYST_CLAIM]),
        _msg(AgentName.CRITIC, AgentName.REPORT, confidence=0.9, verdict=Verdict.PASS.value),
    ]

    verdict, claims = summarize_result({"history": history, "verdict": Verdict.PASS})

    assert verdict == "pass"
    assert claims == [ANALYST_CLAIM]


# ---------------------------------------------------------------------------
# 3. What reaches the report.
# ---------------------------------------------------------------------------


def test_collect_report_claims_includes_spatial_and_biology_with_distinct_ids():
    claims = collect_report_claims(_state(_full_history()))

    assert [c["claim_id"] for c in claims] == ["cell-0", "spatial:cell-0", "biology:GO:0006955"]
    assert len({c["claim_id"] for c in claims}) == 3  # Spatial's cell-0 must not shadow Analyst's

    report = render_report(RUN_ID, TASK_ID, [ReportClaimInput(**c) for c in claims])
    assert [(e.claim_type, e.value) for e in report.entries] == [
        ("cell_type_call", "T cell"),
        ("niche_enrichment", "niche-2"),
        ("pathway_enrichment", "immune response"),
    ]
    html_doc = render_report_html(report)
    assert "niche-2" in html_doc
    assert "immune response" in html_doc


def test_collect_report_claims_ignores_a_vetoed_first_attempt():
    history = [
        _msg(AgentName.ANALYST, AgentName.CRITIC, confidence=0.3, claims=[{**ANALYST_CLAIM, "cell_type": "vetoed"}]),
        _veto(AgentName.ANALYST),
        _msg(AgentName.ANALYST, AgentName.CRITIC, claims=[ANALYST_CLAIM]),
    ]

    claims = collect_report_claims(_state(history))

    assert [c["cell_type"] for c in claims] == ["T cell"]


def test_evidence_claim_id_keeps_analyst_ids_and_namespaces_the_rest():
    assert evidence_claim_id(AgentName.ANALYST, "cell-0") == "cell-0"
    assert evidence_claim_id(AgentName.SPATIAL, "cell-0") == "spatial:cell-0"
    assert evidence_claim_id(AgentName.BIOLOGY, "GO:1") == "biology:GO:1"


# ---------------------------------------------------------------------------
# 4. Evidence for every claim producer, under the ids the report looks up.
# ---------------------------------------------------------------------------


def test_critic_pass_on_biology_still_stores_evidence_for_analyst_and_spatial(monkeypatch):
    stored = []

    def _fail_to_load(uri):
        raise RuntimeError("object store unreachable in this test")

    monkeypatch.setattr("nexus_agent.data.object_store.get_anndata", _fail_to_load)
    monkeypatch.setattr("nexus_agent.learning.celltyping.latest_promoted_checkpoint_bytes", lambda dsn: None)
    monkeypatch.setattr(
        "nexus_agent.xai.literature_rag.retrieve_citations",
        lambda dsn, query, embedder: SimpleNamespace(citations=[], no_supporting_literature_retrieved=True),
    )
    monkeypatch.setattr(
        "nexus_agent.xai.claims_evidence.store_evidence_bundle",
        lambda dsn, run_id, task_id, bundle, heatmap_array=None: stored.append(bundle),
    )
    state = _state(
        _full_history(),
        expression_uri="s3://fake/expr.h5ad",
        subtask_plan=[AgentName.ANALYST, AgentName.SPATIAL, AgentName.BIOLOGY],
    )

    result = critic_node(state)

    assert result["verdict"] == Verdict.PASS
    assert {b.claim_id: b.claim_type for b in stored} == {
        "cell-0": "cell_type_call",
        "spatial:cell-0": "niche_enrichment",
        "biology:GO:0006955": "pathway_enrichment",
    }
    # Same ids the report node will fetch by -- otherwise evidence exists but is never shown.
    report_ids = {c["claim_id"] for c in collect_report_claims(state)}
    assert report_ids == {b.claim_id for b in stored}
    assert all(b.artifacts_present == ["citation"] for b in stored)


# ---------------------------------------------------------------------------
# 5. Veto cap.
# ---------------------------------------------------------------------------


def _history_with_prior_vetoes(n_vetoes: int, confidence: float) -> list[MessageEnvelope]:
    history = []
    for _ in range(n_vetoes):
        history.append(_msg(AgentName.ANALYST, AgentName.CRITIC, confidence=confidence))
        history.append(_veto(AgentName.ANALYST))
    history.append(_msg(AgentName.ANALYST, AgentName.CRITIC, confidence=confidence))
    return history


def test_critic_still_vetoes_below_the_cap():
    in_veto_band = VETO_CONFIDENCE_THRESHOLD - 0.1
    result = critic_node(_state(_history_with_prior_vetoes(MAX_VETOES_PER_AGENT - 1, in_veto_band)))

    assert result["verdict"] == Verdict.VETO


def test_critic_escalates_instead_of_vetoing_past_the_cap():
    in_veto_band = VETO_CONFIDENCE_THRESHOLD - 0.1
    result = critic_node(_state(_history_with_prior_vetoes(MAX_VETOES_PER_AGENT, in_veto_band)))

    envelope = result["history"][0]
    assert result["verdict"] == Verdict.ESCALATE
    assert envelope.to_agent == AgentName.HUMAN_REVIEW
    assert any("cap" in line for line in envelope.payload["reasoning"])


def test_vetoes_of_other_agents_do_not_count_toward_the_cap():
    in_veto_band = VETO_CONFIDENCE_THRESHOLD - 0.1
    history = [_veto(AgentName.VISION)] * MAX_VETOES_PER_AGENT
    history.append(_msg(AgentName.ANALYST, AgentName.CRITIC, confidence=in_veto_band))

    assert critic_node(_state(history))["verdict"] == Verdict.VETO


def test_agent_stuck_in_veto_band_ends_in_human_review_not_a_recursion_error(monkeypatch):
    in_veto_band = VETO_CONFIDENCE_THRESHOLD - 0.1

    def stubborn_analyst(state):
        # What a deterministic real-path agent does on retry: the same answer.
        envelope = MessageEnvelope(
            run_id=state["run_id"],
            task_id=state["task_id"],
            from_agent=AgentName.ANALYST,
            to_agent=AgentName.CRITIC,
            payload={"claims": []},
            confidence=in_veto_band,
            trace_id=state["trace_id"],
        )
        return {"history": [envelope]}

    monkeypatch.setattr(graph_build, "analyst_node", stubborn_analyst)
    graph = compile_with_memory()
    run_id = uuid.uuid4()

    result = graph.invoke(
        dict(
            run_id=run_id,
            task_id=uuid.uuid4(),
            trace_id=uuid.uuid4(),
            goal=AnalyticalGoal(sample_id="sample-1", modalities=["expression"]),
            history=[],
            verdict=None,
            force_verdict=None,
        ),
        config={"configurable": {"thread_id": str(run_id)}},
    )

    critic_verdicts = [m.payload["verdict"] for m in result["history"] if m.from_agent == AgentName.CRITIC]
    assert critic_verdicts == ["veto"] * MAX_VETOES_PER_AGENT + ["escalate"]
    assert result["verdict"] == Verdict.ESCALATE
    assert result["history"][-1].to_agent == AgentName.HUMAN_REVIEW


# ---------------------------------------------------------------------------
# 6. Best-effort failures leave a trace.
# ---------------------------------------------------------------------------


def test_message_log_failure_is_logged_and_does_not_block_the_node(monkeypatch, caplog):
    def _unreachable(*args, **kwargs):
        raise OSError("postgres is down")

    monkeypatch.setattr("psycopg.connect", _unreachable)
    envelope = _msg(AgentName.ANALYST, AgentName.CRITIC)
    wrapped = _with_message_logging(lambda state: {"history": [envelope]})

    with caplog.at_level(logging.WARNING, logger="nexus_agent.graph.build"):
        result = wrapped(_state([], expression_uri="s3://fake/expr.h5ad"))

    assert result == {"history": [envelope]}
    assert "message_log" in caplog.text
    assert "postgres is down" in caplog.text


def test_escalation_persistence_failure_is_logged(monkeypatch, caplog):
    def _unreachable(dsn, **kwargs):
        raise OSError("postgres is down")

    monkeypatch.setattr("nexus_agent.review.escalation.record_escalation", _unreachable)
    qc = _msg(AgentName.QC, AgentName.HUMAN_REVIEW, confidence=0.1)

    with caplog.at_level(logging.WARNING, logger="nexus_agent.graph.build"):
        assert _human_review_node(_state([qc], image_uri="s3://fake/image.npy")) == {}

    assert "could not record escalation" in caplog.text


@pytest.mark.parametrize("uris", [{}, {"image_uri": None, "expression_uri": None}])
def test_stub_path_human_review_stays_a_pure_noop(monkeypatch, uris):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("stub path must not touch Postgres")

    monkeypatch.setattr("nexus_agent.review.escalation.record_escalation", _must_not_be_called)
    qc = _msg(AgentName.QC, AgentName.HUMAN_REVIEW, confidence=0.1)

    assert _human_review_node(_state([qc], **uris)) == {}
