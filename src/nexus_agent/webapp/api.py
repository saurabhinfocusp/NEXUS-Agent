"""The user-facing FastAPI app: upload a sample, poll for its report.

Composes three things under one origin so the static frontend
(`webapp/static/index.html`) only ever talks to one server:

1. `POST /api/runs` / `GET /api/runs/{run_id}` -- submit a sample (image +
   expression file), poll its status. Kicks the real pipeline off as a
   `BackgroundTasks` job (`webapp/pipeline_runner.py`) since real CellPose/
   VGG16 inference takes minutes on CPU and must not block the request that
   started it. This is a single-process, in-memory-scheduled background
   task -- fine for a dev/small-scale deployment, not a substitute for a
   real job queue at the Kubernetes-batch-job scale Art. XII §8 eventually
   targets.
2. `GET /api/artifacts/heatmap` -- a small proxy that turns a stored
   heatmap's `s3://` URI into a real `image/png` a browser can fetch.
   `report/build.py::render_report_html` (Phase 4, unmodified here) embeds
   `<img src="{heatmap_uri}">` with the raw object-store URI, which no
   browser can load directly -- `GET /api/runs/{run_id}` rewrites those
   occurrences in the stored `report_html` to point at this proxy instead.
3. `/review/*` -- `review/api.py`'s existing corrections/escalations/metrics
   app (Phase 5, unmodified), mounted as a sub-application.

The static-file mount for `/` is registered LAST: Starlette matches routes
in registration order, so a `/` mount registered earlier would swallow
every route defined after it.
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from nexus_agent.data.object_store import get_array, put_anndata, put_array
from nexus_agent.review.api import app as review_app
from nexus_agent.shared.config import settings
from nexus_agent.webapp.pipeline_runner import run_pipeline_job
from nexus_agent.webapp.runs import PipelineRun, create_run, fetch_run
from nexus_agent.webapp.uploads import read_expression_upload, read_image_upload

_STATIC_DIR = Path(__file__).parent / "static"
_HEATMAP_URI_PATTERN = re.compile(r'src="(s3://[^"]+/heatmap\.npy)"')


class _RevalidateStaticFiles(StaticFiles):
    """`StaticFiles` sets no `Cache-Control` header, so browsers fall back to
    RFC 7234 heuristic caching -- for `index.html`/`app.js`/`styles.css`,
    which change often during dev and carry no version/hash in their URL,
    that can silently serve a stale copy for hours with no request ever
    reaching the server, not even a conditional one. `no-cache` forces a
    revalidation (a fast, small `If-None-Match` round trip -- ETag/
    Last-Modified already do the rest) on every load instead.
    """

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="NEXUS-Agent")
app.mount("/review", review_app)


def _proxy_heatmap_uris(report_html: str) -> str:
    """Rewrite `src="s3://.../heatmap.npy"` to the `/api/artifacts/heatmap`
    proxy so a browser can actually load it, without touching Phase 4's
    `report/build.py`.
    """
    return _HEATMAP_URI_PATTERN.sub(lambda m: f'src="/api/artifacts/heatmap?uri={m.group(1)}"', report_html)


@app.post("/api/runs", status_code=202)
async def create_pipeline_run(
    background_tasks: BackgroundTasks,
    sample_id: str = Form(...),
    image: UploadFile = File(...),
    # A single `.h5ad`/`.loom`/`.h5`/archive, or several loose 10x mtx
    # bundle components (matrix.mtx + barcodes.tsv + features.tsv)
    # selected together -- see webapp/uploads.py::read_expression_upload.
    expression: list[UploadFile] = File(...),
) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="image file is missing a filename")
    if not expression or not all(f.filename for f in expression):
        raise HTTPException(status_code=400, detail="expression file(s) missing a filename")

    image_array = read_image_upload(await image.read(), image.filename)
    try:
        expression_files = [(f.filename, await f.read()) for f in expression]
        adata = read_expression_upload(expression_files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = uuid.uuid4()
    task_id = uuid.uuid4()
    image_uri = put_array(f"webapp/{run_id}/image.npy", image_array)
    expression_uri = put_anndata(f"webapp/{run_id}/expression.h5ad", adata)

    create_run(
        settings.postgres_dsn,
        run_id=run_id,
        task_id=task_id,
        sample_id=sample_id,
        image_uri=image_uri,
        expression_uri=expression_uri,
    )
    background_tasks.add_task(
        run_pipeline_job,
        settings.postgres_dsn,
        run_id=run_id,
        task_id=task_id,
        sample_id=sample_id,
        image_uri=image_uri,
        expression_uri=expression_uri,
    )
    return {"run_id": str(run_id), "status": "pending"}


@app.get("/api/runs/{run_id}")
def get_pipeline_run(run_id: uuid.UUID) -> dict:
    run = fetch_run(settings.postgres_dsn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    payload = run.model_dump(mode="json")
    if run.report_html:
        payload["report_html"] = _proxy_heatmap_uris(run.report_html)
    return payload


@app.get("/api/artifacts/heatmap")
def get_heatmap(uri: str) -> Response:
    heatmap = get_array(uri)
    normalized = heatmap - heatmap.min()
    max_value = normalized.max()
    if max_value > 0:
        normalized = normalized / max_value
    image = Image.fromarray((normalized * 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


app.mount("/", _RevalidateStaticFiles(directory=_STATIC_DIR, html=True), name="static")


__all__ = ["app", "PipelineRun"]
