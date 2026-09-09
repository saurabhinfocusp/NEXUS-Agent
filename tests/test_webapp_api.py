"""Tests for the upload-and-view frontend's FastAPI surface
(src/nexus_agent/webapp/api.py).

The first test is fast and needs no infra -- it just verifies the app is
constructible and routed correctly (including the mounted `/review` sub-app
and the static `/` mount). The full-flow test is `pytest.mark.integration`
and genuinely slow: it runs the real pipeline (real CellPose/VGG16
inference measured ~5 minutes on a CPU-only box), same cost as
tests/test_aji_benchmark.py already accepts -- not shortened, since there's
nothing left to fake once the endpoint is actually invoking `graph.invoke()`.
"""

import io
import time
import uuid

import numpy as np
import pandas as pd
import psycopg
import pytest
from anndata import AnnData
from fastapi.routing import APIRoute, Mount
from fastapi.testclient import TestClient
from PIL import Image
from skimage.draw import disk

from nexus_agent.data.object_store import ensure_bucket
from nexus_agent.shared.config import settings
from nexus_agent.webapp.api import app


def test_app_routes_exist():
    api_paths = {r.path for r in app.routes if isinstance(r, APIRoute)}
    assert "/api/runs" in api_paths
    assert "/api/runs/{run_id}" in api_paths
    assert "/api/artifacts/heatmap" in api_paths

    mount_paths = {r.path for r in app.routes if isinstance(r, Mount)}
    assert "/review" in mount_paths
    assert "" in mount_paths or "/" in mount_paths  # the static-file mount


@pytest.fixture()
def _require_infra():
    try:
        ensure_bucket()
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO/Postgres not reachable ({exc}); run `docker compose up -d`.")


@pytest.fixture()
def client():
    return TestClient(app)


def _synthetic_image_bytes(size: int = 96) -> bytes:
    image = np.zeros((size, size), dtype=np.uint8)
    rr, cc = disk((size // 2, size // 2), 15, shape=image.shape)
    image[rr, cc] = 200
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _synthetic_expression_bytes() -> bytes:
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(0)
    adata = AnnData(
        X=rng.poisson(3, size=(4, 12)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "expression.h5ad"
        adata.write_h5ad(tmp_path)
        return tmp_path.read_bytes()


@pytest.mark.integration
def test_upload_run_and_poll_to_done(_require_infra, client):
    response = client.post(
        "/api/runs",
        data={"sample_id": "webapp-test-sample"},
        files={
            "image": ("image.png", _synthetic_image_bytes(), "image/png"),
            "expression": ("expression.h5ad", _synthetic_expression_bytes(), "application/octet-stream"),
        },
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert response.json()["status"] == "pending"

    # TestClient runs BackgroundTasks synchronously as part of the request
    # above (unlike a real ASGI server, where it runs after the response is
    # sent) -- so by the time we get here the job has already finished, real
    # CellPose/VGG16 inference and all. Poll anyway, bounded, matching the
    # real frontend's behavior rather than assuming synchronous execution.
    deadline = time.monotonic() + 600
    run = None
    while time.monotonic() < deadline:
        poll = client.get(f"/api/runs/{run_id}")
        assert poll.status_code == 200
        run = poll.json()
        if run["status"] in ("done", "failed"):
            break
        time.sleep(2)

    assert run is not None
    assert run["status"] == "done", run.get("error")
    assert run["verdict"] in ("pass", "veto", "escalate")
    assert run["claims"]
    if run["report_html"]:
        assert "/api/artifacts/heatmap?uri=" in run["report_html"] or "s3://" not in run["report_html"]
