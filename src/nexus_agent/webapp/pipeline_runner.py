"""The background job `webapp/api.py::POST /api/runs` schedules.

Runs the real pipeline (Art. III-V) against uploaded image/expression data
and writes its outcome into `pipeline_runs` (`webapp/runs.py`) so the
frontend's poll loop can pick it up. This is FastAPI `BackgroundTasks`
territory -- no exception may propagate out of here (there's no HTTP
request left to raise it to by the time this runs), so failures are
captured into the row via `mark_failed` instead.
"""

from __future__ import annotations

import html
import logging
import uuid

from nexus_agent.agents.common import latest_message_from
from nexus_agent.graph.build import compile_with_postgres
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal
from nexus_agent.webapp.runs import mark_done, mark_failed, mark_running

logger = logging.getLogger(__name__)

# Stored in `pipeline_runs.verdict` when QC's raw-data gate rejected the
# sample before any specialist (or Critic) ran -- Critic never produced a
# verdict on that route, so there is no pass/veto/escalate to report.
QC_FAIL_VERDICT = "qc_fail"


def summarize_result(result: dict) -> tuple[str, list[dict]]:
    """`(verdict, claims)` for `pipeline_runs` from a finished graph run.

    `claims` stays Analyst's per-cell claims only -- it feeds the frontend's
    reviewer-correction form, which is defined over `cell_type`/
    `spatial_domain` (Spatial/Biology findings are shown via `report_html`
    instead) -- and is read from Analyst's *latest* message so a vetoed first
    attempt is never what the reviewer sees.
    """
    analyst_message = latest_message_from(result, AgentName.ANALYST)
    claims = analyst_message.payload.get("claims", []) if analyst_message else []
    if result.get("verdict"):
        verdict = result["verdict"].value
    elif result.get("qc_verdict") == "fail":
        verdict = QC_FAIL_VERDICT
    else:
        verdict = "unknown"
    return verdict, claims


def _apply_question(
    goal: AnalyticalGoal, question: str | None, run_id: uuid.UUID
) -> tuple[AnalyticalGoal, list[str]]:
    """Let the optional LLM intake (`llm/intake.py`) switch on the opt-in
    analyses a free-text question calls for. It can only add to what the
    submitter ticked explicitly, never switch one off, and if it is
    disabled or fails the run proceeds with the explicit flags alone.

    Returns the goal and the parts of the question the pipeline cannot
    answer, which `run_pipeline_job` shows the submitter.
    """
    from nexus_agent.shared.config import settings

    if not (question and question.strip() and settings.llm_intake_enabled):
        return goal, []
    try:
        from nexus_agent.llm.intake import plan_goal

        planned, plan = plan_goal(question, sample_id=goal.sample_id, modalities=goal.modalities)
    except Exception:
        logger.warning("LLM intake failed for run %s; using the submitted flags", run_id, exc_info=True)
        return goal, []
    logger.info("LLM intake for run %s: %s", run_id, plan.rationale)
    goal = goal.model_copy(
        update={
            "run_spatial_analysis": goal.run_spatial_analysis or planned.run_spatial_analysis,
            "run_enrichment_analysis": goal.run_enrichment_analysis or planned.run_enrichment_analysis,
        }
    )
    return goal, plan.out_of_scope


def with_scope_notice(report_html: str | None, out_of_scope: list[str]) -> str | None:
    """Prepend a notice listing the parts of the submitter's question this
    pipeline did not address, so a report that answers half a question
    isn't read as answering all of it. When the run produced no report
    (QC rejection, escalation), the notice stands alone.
    """
    if not out_of_scope:
        return report_html
    items = "".join(f"<li>{html.escape(part)}</li>" for part in out_of_scope)
    notice = (
        '<div class="scope-notice"><p><b>Not addressed by this analysis.</b> '
        f"These parts of your question are outside what the pipeline does:</p><ul>{items}</ul></div>"
    )
    if not report_html:
        return notice
    marker = "<h1>NEXUS-Agent Report</h1>"
    return report_html.replace(marker, marker + "\n" + notice, 1) if marker in report_html else notice + report_html


def run_pipeline_job(
    dsn: str,
    *,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    sample_id: str,
    image_uri: str,
    expression_uri: str,
    run_spatial_analysis: bool = False,
    run_enrichment_analysis: bool = False,
    question: str | None = None,
) -> None:
    mark_running(dsn, run_id)
    try:
        goal = AnalyticalGoal(
            sample_id=sample_id,
            modalities=["image", "expression"],
            run_spatial_analysis=run_spatial_analysis,
            run_enrichment_analysis=run_enrichment_analysis,
        )
        goal, out_of_scope = _apply_question(goal, question, run_id)
        initial_state = {
            "run_id": run_id,
            "task_id": task_id,
            "trace_id": uuid.uuid4(),
            "goal": goal,
            "history": [],
            "verdict": None,
            "force_verdict": None,
            "stub_confidence_override": None,
            "image_uri": image_uri,
            "expression_uri": expression_uri,
        }
        with compile_with_postgres(dsn) as graph:
            result = graph.invoke(initial_state, config={"configurable": {"thread_id": str(run_id)}})

        verdict, claims = summarize_result(result)
        report_html = with_scope_notice(result.get("report_html"), out_of_scope)
        mark_done(dsn, run_id, verdict=verdict, claims=claims, report_html=report_html)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline run %s failed", run_id)
        mark_failed(dsn, run_id, str(exc))
