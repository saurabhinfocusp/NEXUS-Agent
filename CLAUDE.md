# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NEXUS-Agent is a multi-agent spatial biology analysis pipeline (histology
image + spatial expression data in, evidence-backed biological report out).
[CONSTITUTION.md](CONSTITUTION.md) is the binding design spec — when a
choice in this codebase looks unusual, check there before "fixing" it.
[ROADMAP.md](ROADMAP.md) tracks what's real vs. disclosed scope reduction
per phase; Phases 0-5 are implemented, and its "Post-Phase-5 additions"
section covers what landed afterwards (QC/Spatial/Biology agents, the
webapp, CI) plus the list of known open work.

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

CI (`.github/workflows/tests.yml`) runs only the fast suite, on pushes to
`main` and on pull requests. The integration suite is never run in CI —
run it locally against `docker compose up -d` before trusting a change to
anything Postgres/S3/model-backed.

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
Coordinator → QC (real path only) → Vision and/or Analyst (per
`subtask_plan`) → Spatial and/or Biology (opt-in) → Critic → `report` /
`human_review` / back to the vetoed agent. All six routers
(`_coordinator_router`, `_qc_router`, `_vision_router`, `_analyst_router`,
`_spatial_router`, `_critic_router`) are plain functions reading
`RunState` — read them before touching routing, rather than inferring flow
from agent code alone:

- Coordinator's edge picks `state["subtask_plan"][0]` — QC whenever a real
  data URI is set, else Vision/Analyst. The stub path never plans QC.
- QC is a raw-data gate, not a claim producer, so it never goes through
  Critic: `qc_verdict == "fail"` routes straight to `human_review`,
  anything else continues to Vision/Analyst.
- Vision's edge goes to Analyst only if Analyst is also in the plan, else
  straight to Critic — Vision and Analyst never both dead-end separately.
- Analyst's edge goes to Spatial, else Biology, else Critic, depending on
  `AnalyticalGoal.run_spatial_analysis`/`run_enrichment_analysis` (both
  default off); Spatial's edge goes to Biology or Critic the same way.
- Critic's edge is the only path from a claim producer to a terminal node:
  `pass` → `report`, `escalate` → `human_review`, `veto` → back to
  `state["history"][-2].from_agent` (the agent whose output Critic just
  rejected), never to a terminal node directly.

Two consequences of that shape that are easy to get wrong:

- `human_review` has two ways in (Critic's `escalate`, QC's `fail`), and
  only the first has Critic's envelope last in `history`. Use
  `graph/build.py::escalated_message` rather than indexing `history[-2]`.
- Vetoes are capped: after `agents/critic.py::MAX_VETOES_PER_AGENT` vetoes
  of the same agent in one run, Critic escalates instead. Real-path agents
  recompute a deterministic result on retry, so an uncapped veto would loop
  until LangGraph's recursion limit.

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

Every agent module under `agents/` (`coordinator.py`, `qc.py`,
`vision.py`, `analyst.py`, `spatial.py`, `biology.py`, `critic.py`)
follows the same shape: a `*_node(state) ->
dict` function returning a partial `RunState` update (LangGraph merges it),
building its outgoing `MessageEnvelope`s via `agents/common.py
::build_envelope`. `agents/common.py` also holds the stub/retry
conventions shared across agents: `is_retry_after_veto` (true when the
immediately preceding message is Critic vetoing this agent) and
`resolve_stub_confidence` (retry-after-veto confidence beats the
test-only `stub_confidence_override`, which only affects the first
attempt).

Analyst, Spatial and Biology are the claim producers
(`agents/critic.py::CLAIM_PRODUCERS`); all three put their claims under
`payload["claims"]` with the claim's id in `cell_id` (Biology reuses that
key for the pathway/term id). Two helpers in `agents/common.py` exist
because reading those claims naively is wrong:

- `latest_message_from(state, agent)` — after a veto the agent's rejected
  first attempt is still in `history`; a forward `next(...)` scan picks up
  the vetoed claims instead of the retry's.
- `evidence_claim_id(agent, raw_id)` — the key a claim's `xai_evidence`
  bundle is stored and fetched under. Analyst ids are used as-is (the
  correction API and `provenance_log` key on them); Spatial's and Biology's
  are prefixed (`spatial:<id>`, `biology:<id>`) because Spatial's per-cell
  ids can equal Analyst's.

Critic builds evidence for every claim producer's latest claims on a
`pass`, not just the message it reviewed, and
`graph/build.py::collect_report_claims` gathers the same set for the
report — keep those two in step, or evidence gets stored under ids the
report never looks up. Heatmap/SHAP apply to Analyst's per-cell claims
only; Spatial/Biology claims get the literature artifact.

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
   (`webapp/runs.py`) for status. `pipeline_runs.verdict` is Critic's
   verdict, or `qc_fail` when QC rejected the sample before Critic ran
   (`pipeline_runner.py::summarize_result`); `claims` is Analyst's claims
   only, since it feeds the reviewer-correction form — Spatial/Biology
   findings reach the user through `report_html`.
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
everywhere Postgres is touched. `settings.nexus_device` (`NEXUS_DEVICE`,
default `cpu`) selects the torch device for Vision's CellPose and
VGG16/DINOv2 inference only — the fusion transformer, the fine-tune loop
and Grad-CAM++ do not read it yet.

### Best-effort persistence logs, it doesn't raise

Dashboard/evidence/escalation writes (`_with_message_logging`,
`_human_review_node`, Critic's evidence bundles, QC's plot upload) are
deliberately wrapped so a Postgres/S3 failure never fails the run — but
each one logs a `logger.warning(..., exc_info=True)`. Keep that when adding
another best-effort write; a bare `except Exception: pass` here means a
run silently finishes with missing evidence.
