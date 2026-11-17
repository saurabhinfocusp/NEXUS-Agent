"""Tests for the optional LLM layer (nexus_agent/llm/).

No network and no API key: the real `anthropic` SDK is driven against a
mock HTTP transport, so these check the request this code actually sends
and how it handles what comes back. They say nothing about what a real
model would write -- the layer has not been run against the live API.
"""

import json
import uuid

import pytest

anthropic = pytest.importorskip("anthropic")
import httpx2  # noqa: E402  (anthropic's HTTP library; present whenever anthropic is)

from nexus_agent.graph import build as graph_build  # noqa: E402
from nexus_agent.llm.client import FALLBACK_BETA, LLMUnavailable  # noqa: E402
from nexus_agent.llm.intake import plan_goal  # noqa: E402
from nexus_agent.llm.narrative import (  # noqa: E402
    Finding,
    NarrativeDraft,
    NarrativeSentence,
    render_narrative_html,
    summarize_findings,
    verify_narrative,
    write_narrative,
)
from nexus_agent.report.build import ReportClaimInput, render_report, render_report_html  # noqa: E402
from nexus_agent.shared.config import settings  # noqa: E402
from nexus_agent.webapp import pipeline_runner  # noqa: E402
from nexus_agent.webapp.pipeline_runner import with_scope_notice  # noqa: E402


def _client(reply: dict | None = None, *, stop_reason: str = "end_turn", status: int = 200, seen: list | None = None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append((request, json.loads(request.content)))
        if status != 200:
            return httpx2.Response(status, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        content = [] if reply is None else [{"type": "text", "text": json.dumps(reply)}]
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    return anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def _plan(**overrides) -> dict:
    plan = {"run_spatial_analysis": False, "run_enrichment_analysis": False, "rationale": "r", "out_of_scope": []}
    plan.update(overrides)
    return plan


# --- intake --------------------------------------------------------------------


def test_intake_sends_a_structured_request_with_fallbacks_and_explicit_effort():
    seen: list = []
    client = _client(_plan(run_spatial_analysis=True), seen=seen)

    plan_goal("Which cell groups sit next to each other?", sample_id="s1", modalities=["image", "expression"], client=client)

    request, body = seen[0]
    assert request.url.path == "/v1/messages"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["model"] == settings.llm_model == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body and "temperature" not in body
    assert "Which cell groups sit next to each other?" in body["messages"][0]["content"]
    assert "image, expression" in body["messages"][0]["content"]


def test_intake_sets_only_the_opt_in_flags_and_reports_what_is_out_of_scope():
    client = _client(_plan(run_spatial_analysis=True, out_of_scope=["survival analysis"]))

    goal, plan = plan_goal("neighbours, and survival?", sample_id="s1", modalities=["image", "expression"], client=client)

    assert (goal.run_spatial_analysis, goal.run_enrichment_analysis) == (True, False)
    assert goal.sample_id == "s1" and goal.modalities == ["image", "expression"]
    assert plan.out_of_scope == ["survival analysis"]


def test_intake_cannot_request_expression_analyses_without_expression_data():
    client = _client(_plan(run_spatial_analysis=True, run_enrichment_analysis=True))

    goal, _plan_ = plan_goal("pathways?", sample_id="s1", modalities=["image"], client=client)

    assert goal.modalities == ["image"]
    assert not goal.run_spatial_analysis and not goal.run_enrichment_analysis


@pytest.mark.parametrize(
    ("client_kwargs", "message"),
    [
        ({"reply": None, "stop_reason": "refusal"}, "declined"),
        ({"reply": None, "stop_reason": "max_tokens"}, "no complete structured output"),
        ({"status": 500}, "Anthropic API error 500"),
    ],
)
def test_llm_failures_surface_as_llm_unavailable(client_kwargs, message):
    with pytest.raises(LLMUnavailable, match=message):
        plan_goal("q", sample_id="s1", modalities=["expression"], client=_client(**client_kwargs))


# --- narrative: findings and verification ---------------------------------------


def _report():
    claims = [
        ReportClaimInput(claim_id="cell-0", spatial_domain="domain_0", confidence=0.9),
        ReportClaimInput(claim_id="cell-1", spatial_domain="domain_0", confidence=0.8),
        ReportClaimInput(claim_id="cell-2", spatial_domain="domain_1", confidence=0.2),
        ReportClaimInput(claim_id="cell-3", spatial_domain="domain_10", confidence=0.5),
        ReportClaimInput(claim_id="biology:GO:1", pathway="immune response", confidence=0.95),
        ReportClaimInput(claim_id="cell-4", confidence=0.6),  # asserts nothing
    ]
    return render_report(uuid.uuid4(), uuid.uuid4(), claims)


def _findings() -> dict[str, Finding]:
    return {f.value: f for f in summarize_findings(_report())}


def test_findings_group_claims_by_type_and_value_and_skip_valueless_ones():
    findings = _findings()

    assert set(findings) == {"domain_0", "domain_1", "domain_10", "immune response"}
    domain_0 = findings["domain_0"]
    assert domain_0.n_claims == 2
    assert domain_0.mean_confidence == pytest.approx(0.85) and domain_0.min_confidence == pytest.approx(0.8)
    assert domain_0.tier == "high"
    assert findings["domain_1"].tier == "low" and findings["domain_1"].share_low_confidence == 1.0
    assert len({f.finding_id for f in findings.values()}) == 4


def _verify(*sentences: tuple[str, list[str]]):
    draft = NarrativeDraft(sentences=[NarrativeSentence(text=t, finding_ids=ids) for t, ids in sentences])
    return verify_narrative(draft, list(_findings().values()))


def test_grounded_sentence_is_kept_at_the_tier_of_its_weakest_finding():
    f = _findings()
    strong, weak = f["domain_0"].finding_id, f["domain_1"].finding_id

    narrative = _verify(
        ("Most cells fall in domain_0.", [strong]),
        ("domain_0 is larger than domain_1.", [strong, weak]),
    )

    assert [s.tier for s in narrative.sentences] == ["high", "low"]
    assert narrative.rejected == []


@pytest.mark.parametrize(
    ("text", "ids", "reason"),
    [
        ("The tissue is inflamed.", [], "cites no finding"),
        ("Most cells fall in domain_0.", ["F99"], "not in the report"),
        # cites domain_0's finding but talks about the pathway result
        ("domain_0 shows an immune response.", ["DOMAIN_0"], "immune response"),
    ],
)
def test_ungrounded_sentences_are_rejected_with_the_reason(text, ids, reason):
    f = _findings()
    ids = [f["domain_0"].finding_id if i == "DOMAIN_0" else i for i in ids]

    narrative = _verify((text, ids))

    assert narrative.sentences == []
    assert reason in narrative.rejected[0].reason


def test_value_matching_respects_term_boundaries_and_case():
    f = _findings()

    # "domain_10" does not mention domain_1; "Immune Response" does mention the pathway
    assert len(_verify(("domain_10 is small.", [f["domain_10"].finding_id])).sentences) == 1
    assert len(_verify(("An Immune Response signature.", [f["domain_0"].finding_id])).rejected) == 1


def test_write_narrative_verifies_what_the_model_returns():
    f = _findings()
    seen: list = []
    reply = {
        "sentences": [
            {"text": "Most cells fall in domain_0.", "finding_ids": [f["domain_0"].finding_id]},
            {"text": "This indicates a tumour margin.", "finding_ids": []},
        ]
    }

    narrative = write_narrative(_report(), client=_client(reply, seen=seen))

    assert [s.text for s in narrative.sentences] == ["Most cells fall in domain_0."]
    assert [r.reason for r in narrative.rejected] == ["cites no finding"]
    _request, body = seen[0]
    assert body["output_config"]["effort"] == "medium"
    assert "domain_0" in body["messages"][0]["content"]
    assert "cell-0" not in body["messages"][0]["content"]  # findings, not per-cell claims, are sent


def test_write_narrative_makes_no_request_when_there_is_nothing_to_summarize():
    report = render_report(uuid.uuid4(), uuid.uuid4(), [ReportClaimInput(claim_id="cell-0", confidence=0.6)])

    assert write_narrative(report, client=_client(status=500)) is None


def test_rendered_narrative_is_tier_styled_escaped_and_discloses_removals():
    f = _findings()
    narrative = _verify(
        ("Cells in domain_1 <b>are few</b>.", [f["domain_1"].finding_id]),
        ("Unsupported.", []),
    )

    section = render_narrative_html(narrative)

    assert 'class="narrative-sentence tier-low"' in section
    assert "&lt;b&gt;are few&lt;/b&gt;" in section and "<b>are few</b>" not in section
    assert "Written by a language model" in section
    assert "1 drafted sentence(s) were removed" in section
    assert section in render_report_html(_report(), narrative_html=section)


# --- wiring: off by default, never fatal ----------------------------------------


def test_report_has_no_narrative_and_makes_no_llm_call_by_default(monkeypatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("LLM narrative must not run unless enabled")

    monkeypatch.setattr("nexus_agent.llm.narrative.write_narrative", _must_not_be_called)

    assert settings.llm_narrative_enabled is False
    assert graph_build._narrative_html(_report()) == ""


def test_narrative_failure_leaves_the_report_without_one(monkeypatch, caplog):
    def _unavailable(report, **kwargs):
        raise LLMUnavailable("no credentials")

    monkeypatch.setattr(settings, "llm_narrative_enabled", True)
    monkeypatch.setattr("nexus_agent.llm.narrative.write_narrative", _unavailable)

    with caplog.at_level("WARNING", logger="nexus_agent.graph.build"):
        assert graph_build._narrative_html(_report()) == ""
    assert "could not write the report narrative" in caplog.text


def _goal(**flags):
    from nexus_agent.shared.schemas import AnalyticalGoal

    return AnalyticalGoal(sample_id="s1", modalities=["image", "expression"], **flags)


def test_question_is_ignored_when_intake_is_disabled(monkeypatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("LLM intake must not run unless enabled")

    monkeypatch.setattr("nexus_agent.llm.intake.plan_goal", _must_not_be_called)

    assert pipeline_runner._apply_question(_goal(), "pathways?", uuid.uuid4()) == (_goal(), [])


def test_intake_can_add_analyses_but_never_remove_an_explicit_one(monkeypatch):
    from nexus_agent.llm.intake import GoalPlan

    planned = _goal(run_enrichment_analysis=True)  # the model asks for enrichment only
    plan = GoalPlan(**_plan(run_enrichment_analysis=True, out_of_scope=["survival analysis"]))
    monkeypatch.setattr(settings, "llm_intake_enabled", True)
    monkeypatch.setattr("nexus_agent.llm.intake.plan_goal", lambda *a, **k: (planned, plan))

    goal, out_of_scope = pipeline_runner._apply_question(
        _goal(run_spatial_analysis=True), "pathways and survival?", uuid.uuid4()
    )

    assert goal.run_spatial_analysis and goal.run_enrichment_analysis
    assert out_of_scope == ["survival analysis"]


def test_intake_failure_falls_back_to_the_submitted_flags(monkeypatch):
    def _unavailable(*args, **kwargs):
        raise LLMUnavailable("no credentials")

    monkeypatch.setattr(settings, "llm_intake_enabled", True)
    monkeypatch.setattr("nexus_agent.llm.intake.plan_goal", _unavailable)
    submitted = _goal(run_spatial_analysis=True)

    assert pipeline_runner._apply_question(submitted, "pathways?", uuid.uuid4()) == (submitted, [])


def test_scope_notice_is_shown_with_or_without_a_report():
    report_html = render_report_html(_report())

    with_report = with_scope_notice(report_html, ["survival <analysis>"])
    alone = with_scope_notice(None, ["survival <analysis>"])

    assert "Not addressed by this analysis" in with_report and "survival &lt;analysis&gt;" in with_report
    assert with_report.index("NEXUS-Agent Report</h1>") < with_report.index("scope-notice")
    assert "claim-entry" in with_report  # the report itself is intact
    assert "Not addressed by this analysis" in alone
    assert with_scope_notice(report_html, []) == report_html
