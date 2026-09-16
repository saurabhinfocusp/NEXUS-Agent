# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NEXUS-Agent is a multi-agent spatial biology analysis pipeline (histology
image + spatial expression data in, evidence-backed biological report out).
[CONSTITUTION.md](CONSTITUTION.md) is the binding design spec — when a
choice in this codebase looks unusual, check there before "fixing" it.
[ROADMAP.md](ROADMAP.md) tracks what's real vs. disclosed scope reduction
per phase; Phases 0-5 are implemented.

## Commands

```bash
# Environment (installs the project editable + the Phase 3 model stack).
# torch/torchvision are pulled from PyTorch's CPU wheel index FIRST inside
# environment.yml's single `pip:` block -- installing the project without
# that ordering pulls PyPI's default CUDA-bundled torch wheel instead.
conda env create -f environment.yml
conda activate nexus-agent

# Local infra: Postgres+pgvector, Redis, MinIO
cp .env.example .env
docker compose up -d

# Fast tests (no Docker, no model weights)
pytest tests/ -m "not integration" -q

# Integration tests (Docker + real CellPose/VGG16 inference + Postgres/pgvector)
pytest tests/ -m integration

# Single test
pytest tests/test_critic.py::test_veto_below_threshold -q

# Upload-and-view webapp (FastAPI + static frontend, one origin)
uvicorn nexus_agent.webapp.api:app --reload

# Reviewer API standalone (escalations/corrections/metrics; also mounted
# at /review under the webapp above)
uvicorn nexus_agent.review.api:app --reload

# Fetch one real HEST-1k sample (image + .h5ad) to exercise the real
# pipeline path instead of the synthetic stub (needs a HF token, see the
# script's docstring)
python scripts/download_hest_sample.py TENX95
```

There is no CLI and no lint/format tooling configured — the graph is
invoked directly from Python (see README.md's "Running a task") or through
the webapp's HTTP surface.

`--reload`'s graceful shutdown waits for in-flight `BackgroundTasks`
before restarting -- editing a `.py` file while a real pipeline run is
executing (`webapp/pipeline_runner.py::run_pipeline_job`, real CPU
inference, can take many minutes) leaves the dev server unresponsive to
all requests, including that run's own status polling, until it finishes
or the process is force-killed. Not a bug, just a real trap when iterating
on `webapp/` against a real upload instead of the fast synthetic-stub path.

## Architecture

### The graph is the source of truth for control flow

`graph/build.py::build_graph()` assembles a LangGraph `StateGraph`:
Coordinator → (Vision and/or Analyst, per `subtask_plan`) → Critic →
`report` / `human_review` / back to the vetoed agent. All three routers
(`_coordinator_router`, `_vision_router`, `_critic_router`) are plain
functions reading `RunState` — read them before touching routing, rather
than inferring flow from agent code alone:

- Coordinator's edge picks `state["subtask_plan"][0]`.
- Vision's edge goes to Analyst only if Analyst is also in the plan, else
  straight to Critic — Vision and Analyst never both dead-end separately.
- Critic's edge is the only path to a terminal node: `pass` → `report`,
  `escalate` → `human_review`, `veto` → back to `state["history"][-2]
  .from_agent` (the agent whose output Critic just rejected), never to a
  terminal node directly.

Two nodes (`_report_node`, `_human_review_node`) and the
`_with_message_logging` wrapper around every agent node all gate real work
behind `state.get("image_uri") or state.get("expression_uri")` — the
Phase 0/1 synthetic-stub path (no image/expression URIs) must stay a
pure in-memory no-op so unit tests never need Postgres. When adding a new
node or wrapper that touches Postgres/S3, follow this same gate.

### Agents communicate through graph state, not through the Redis bus

`bus/client.py`'s `MessageBus` validates and publishes `MessageEnvelope`s
to Redis Streams (Constitution Art. XII §4) and is fully implemented — but
nothing in `graph/` or `agents/` actually calls it. Inter-agent messages
flow as `MessageEnvelope` objects appended to `state["history"]` inside
the LangGraph run; the Redis bus is a validated-envelope primitive that
exists independently. Don't assume a message crossing agents implies a
Redis round-trip.

Every agent module under `agents/` (`coordinator.py`, `vision.py`,
`analyst.py`, `critic.py`) follows the same shape: a `*_node(state) ->
dict` function returning a partial `RunState` update (LangGraph merges it),
building its outgoing `MessageEnvelope`s via `agents/common.py
::build_envelope`. `agents/common.py` also holds the stub/retry
conventions shared across agents: `is_retry_after_veto` (true when the
immediately preceding message is Critic vetoing this agent) and
`resolve_stub_confidence` (retry-after-veto confidence beats the
test-only `stub_confidence_override`, which only affects the first
attempt).

### Real vs. stub dual path

Most nodes have two code paths selected by whether `image_uri`/
`expression_uri` are set on state: a Phase 0/1 synthetic single-cell stub
(no external calls, what the fast test suite exercises), and the real
path (CellPose segmentation + VGG16 embedding in `vision/`, graph
transformer fusion in `analyst/`, Grad-CAM++/SHAP/RAG evidence in `xai/`).
When changing a node's behavior, check which path a given test is
actually exercising — `tests/test_graph_real_pipeline.py` is the
canonical example of driving the real path end-to-end.

### Postgres tables and who owns them

`db/schema.sql` defines everything Postgres-backed *except* LangGraph's
own checkpoint tables, which `PostgresSaver.setup()` (called inside
`graph/build.py::compile_with_postgres`) migrates itself — don't add
LangGraph state tables to `db/schema.sql`. What's there: `provenance_log`
(per-claim fusion provenance, Phase 4/5), `correction_log` (Phase 5
reviewer corrections, feeds `learning/lora_finetune.py`), `literature_chunks`
(pgvector RAG corpus), `xai_evidence` (Critic's per-claim evidence, stored
independently of rendered report text), `message_log` (dashboard
substrate, written by `_with_message_logging`), `escalation_queue`
(Phase 5 human-review queue), `finetune_runs` (fine-tune/promotion-gate
log), and `pipeline_runs` (the webapp's async run tracker).

### webapp/ composes three surfaces under one FastAPI origin

`webapp/api.py` is the upload-and-view frontend's backend, and it exists
so the static frontend only ever talks to one server:

1. `POST /api/runs` / `GET /api/runs/{run_id}` — submits a sample, kicks
   the real pipeline off via FastAPI `BackgroundTasks`
   (`webapp/pipeline_runner.py`) since real CPU inference takes minutes
   and must not block the request, and polls `pipeline_runs`
   (`webapp/runs.py`) for status.
2. `GET /api/artifacts/heatmap` — proxies a stored heatmap's `s3://` URI
   into a real `image/png`, since `report/build.py`'s rendered
   `report_html` embeds the raw object-store URI no browser can load;
   `_proxy_heatmap_uris` rewrites those occurrences before the frontend
   ever sees them.
3. `/review/*` — `review/api.py`'s app (escalations, corrections,
   `/metrics`), mounted as a sub-application, unmodified.

The static-file mount (`webapp/static/`: `index.html` + `styles.css` +
`app.js`, no build step) is registered **last** — Starlette matches
routes in registration order, so mounting `/` earlier would swallow every
API route defined after it. `review/api.py::get_metrics` lazily imports
`nexus_agent.metrics.dashboards` and returns 503 if it's not
importable — that module can legitimately not exist yet.

`webapp/uploads.py::read_expression_upload` takes `list[tuple[filename,
bytes]]`, not just an `.h5ad` — a single file dispatches on extension
(`.h5ad`, `.loom`, a 10x `.h5` matrix, or a `.zip`/`.tar`/`.tar.gz`/`.tgz`
archive, itself falling back from "is it a 10x mtx bundle" to "does it
just wrap one recognizable file"), while 2+ files are assumed to be a 10x
mtx bundle's loose components (`matrix.mtx`/`barcodes.tsv`/
`features.tsv`, each optionally `.gz`). Any single file can also be
plain-gzipped (`_maybe_gunzip`, applied before the rest of the dispatch).
`.rds` and `.cloupe` are deliberately rejected with a specific message
pointing at a real conversion path rather than attempted — see that
module's docstring before adding a new format.

Same rationale, applied to pipeline *execution* rather than upload
*format*: the QC and Spatial Transcriptomics agents (`agents/qc.py`,
`agents/spatial.py`) intentionally do not execute FastQC, STARsolo, Cell
Ranger, Seurat, or Giotto. FastQC/STARsolo/Cell Ranger are FASTQ→counts
pipeline tools, and this repo never ingests raw FASTQ (only images and
count matrices/`.h5ad`); Seurat and Giotto are R packages, and the project
has the same deliberate no-R-interop stance that already rejects `.rds`
uploads above. QC and Spatial instead implement the Python-native subset
with real logic: Scanpy `sc.pp.calculate_qc_metrics` + scikit-image
Laplacian-variance/Otsu QC in place of FastQC/STARsolo/Cell Ranger's own
QC output, and Squidpy `nhood_enrichment`/`co_occurrence`/`spatial_autocorr`
in place of Seurat/Giotto's spatial-domain analysis.

The frontend tracks multiple runs independently, not one at a time:
`app.js`'s `pollHandles` map holds one `setInterval` per run ID, each
polling and updating only its own DOM card, and `localStorage` caches
every run's last-known status (`nexus.recentRuns`) so a reload shows
history instantly without re-fetching finished runs. Submitting a new run
never touches another run's polling.

### Config

`shared/config.py::settings` is a single `pydantic-settings` `Settings`
instance, `.env`-backed (see `.env.example`), consumed directly by
whatever module needs a DSN/endpoint — there's no dependency injection
layer. `settings.postgres_dsn` is the assembled connection string used
everywhere Postgres is touched.
