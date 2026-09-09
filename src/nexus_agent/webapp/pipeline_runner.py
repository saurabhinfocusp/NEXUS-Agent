"""The background job `webapp/api.py::POST /api/runs` schedules.

Runs the real pipeline (Art. III-V) against uploaded image/expression data
and writes its outcome into `pipeline_runs` (`webapp/runs.py`) so the
frontend's poll loop can pick it up. This is FastAPI `BackgroundTasks`
territory -- no exception may propagate out of here (there's no HTTP
request left to raise it to by the time this runs), so failures are
captured into the row via `mark_failed` instead.
"""

from __future__ import annotations

import uuid

from nexus_agent.graph.build import compile_with_postgres
from nexus_agent.shared.schemas import AgentName, AnalyticalGoal
from nexus_agent.webapp.runs import mark_done, mark_failed, mark_running


def run_pipeline_job(
    dsn: str,
    *,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    sample_id: str,
    image_uri: str,
    expression_uri: str,
) -> None:
    mark_running(dsn, run_id)
    try:
        initial_state = {
            "run_id": run_id,
            "task_id": task_id,
            "trace_id": uuid.uuid4(),
            "goal": AnalyticalGoal(sample_id=sample_id, modalities=["image", "expression"]),
            "history": [],
            "verdict": None,
            "force_verdict": None,
            "stub_confidence_override": None,
            "image_uri": image_uri,
            "expression_uri": expression_uri,
        }
        with compile_with_postgres(dsn) as graph:
            result = graph.invoke(initial_state, config={"configurable": {"thread_id": str(run_id)}})

        analyst_message = next((m for m in result["history"] if m.from_agent == AgentName.ANALYST), None)
        claims = analyst_message.payload.get("claims", []) if analyst_message else []
        verdict = result["verdict"].value if result.get("verdict") else "unknown"
        mark_done(dsn, run_id, verdict=verdict, claims=claims, report_html=result.get("report_html"))
    except Exception as exc:  # noqa: BLE001
        mark_failed(dsn, run_id, str(exc))
