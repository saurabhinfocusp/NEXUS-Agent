"""Reviewer-facing FastAPI surface (Constitution Art. VII §1, §3).

Exposes the escalation queue and correction interface over HTTP -- the
actual "reviewer-facing surface" Art. VII §3 calls for, instead of a
terminating graph edge. Also exposes a `/metrics` Prometheus-exposition
endpoint that lazily delegates to `nexus_agent.metrics.dashboards`, since
that module is developed independently and may not exist yet at any given
time this file is imported or tested.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from nexus_agent.review.correction import (
    CorrectionRequest,
    fetch_corrections,
    submit_correction,
)
from nexus_agent.review.escalation import list_pending_escalations, resolve_escalation
from nexus_agent.shared.config import settings

app = FastAPI(title="NEXUS-Agent Review API")


@app.get("/escalations")
def get_escalations() -> list[dict]:
    records = list_pending_escalations(settings.postgres_dsn)
    return [r.model_dump(mode="json") for r in records]


@app.post("/escalations/{escalation_id}/resolve")
def post_resolve_escalation(escalation_id: int) -> dict:
    resolve_escalation(settings.postgres_dsn, escalation_id)
    return {"status": "resolved", "id": escalation_id}


@app.post("/corrections", status_code=201)
def post_correction(request: CorrectionRequest) -> JSONResponse:
    new_id = submit_correction(settings.postgres_dsn, request)
    return JSONResponse(status_code=201, content={"id": new_id})


@app.get("/corrections")
def get_corrections(
    since: datetime | None = None,
    run_id: uuid.UUID | None = None,
) -> list[dict]:
    records = fetch_corrections(settings.postgres_dsn, since=since, run_id=run_id)
    return [r.model_dump(mode="json") for r in records]


@app.get("/metrics")
def get_metrics() -> PlainTextResponse:
    try:
        from nexus_agent.metrics.dashboards import prometheus_exposition
    except (ImportError, AttributeError):
        return PlainTextResponse(
            content="metrics module not yet available",
            status_code=503,
        )
    return PlainTextResponse(
        content=prometheus_exposition(settings.postgres_dsn),
        media_type="text/plain",
    )
