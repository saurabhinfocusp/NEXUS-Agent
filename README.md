# NEXUS-Agent

**Next-Generation Explainable Unified Spatial-Agent**
Infocusp · Lead: Saurabh Gupta

A unified, explainable, multi-agent AI system that executes the complete
spatial biology analysis pipeline — from raw histology/multiplex images and
spatial sequencing data to an evidence-backed, human-verifiable biological
report.

- [CONSTITUTION.md](CONSTITUTION.md) — the project's founding charter:
  mission, agent architecture, evidentiary standards, and the technical
  reference spec every implementation decision is bound by.
- [ROADMAP.md](ROADMAP.md) — the phase-by-phase work plan derived from the
  Constitution, with implementation steps and exit criteria per phase.
  **Current status: Phases 0–5 implemented** (see ROADMAP.md's per-phase
  status notes for what's real vs. disclosed scope reductions).

## Architecture at a glance

A LangGraph `StateGraph` routes work between seven specialist agents —
Coordinator, QC, Vision, Analyst, Spatial Transcriptomics, Biology
(Enrichment), Critic/XAI — communicating via schema-validated messages over
a Redis Streams bus, with every biological claim in the final report
required to carry a visual attribution map, a feature-importance score,
and/or a literature citation (see Constitution Art. III and VI). QC runs
first as a raw-data quality gate (Scanpy/scikit-image metrics), ahead of
Vision/Analyst, and escalates straight to human review on failure rather
than passing a claim through Critic. Spatial Transcriptomics and Biology
are optional specialists that run after Analyst — spatial-neighborhood
biology via Squidpy, and gene-set/pathway enrichment via gseapy/g:Profiler/
STRING — and, like Vision/Analyst, their claims are always reviewed by
Critic before reaching the report. Critic generates that evidence for real
(Grad-CAM++, SHAP, a pgvector-backed RAG literature layer) and persists it
independently of the report; a FastAPI reviewer surface lets an expert
correct a claim, which feeds a real PEFT/LoRA fine-tuning loop gated by a
promotion regression-check before a new checkpoint ever reaches a live
prediction. Full architectural and constitutional detail lives in the two
docs linked above; this README only covers running what's built so far.

## Quickstart

```bash
# 1. Create the conda environment (installs the project in editable mode,
#    including the Phase 3 model stack -- torch/torchvision from the CPU
#    wheel index, cellpose, scanpy, squidpy -- plus the Phase 4/5 XAI and
#    learning-loop stack: shap, sentence-transformers, scikit-learn,
#    pgvector, peft, fastapi, uvicorn. ~3GB total including first-run
#    model weight downloads.)
conda env create -f environment.yml
conda activate nexus-agent

# 2. Bring up local infra: Postgres (with the pgvector extension, for
#    Phase 4's RAG literature index), Redis, MinIO (S3-compatible storage)
cp .env.example .env   # adjust if needed
docker compose up -d

# 3. Run the tests
pytest tests/ -m "not integration" -q   # fast tests (no Docker, no model weights)
pytest tests/ -m integration            # integration tests: Docker + real
                                         # CellPose/VGG16 inference + a one-time
                                         # ~27MB dataset fetch for the AJI
                                         # benchmark (test_aji_benchmark.py
                                         # alone takes ~15-20 min on CPU) +
                                         # Postgres/pgvector-backed XAI
                                         # evidence, correction/fine-tune,
                                         # and dashboard tests.
```

## Upload-and-view webapp

A small FastAPI + vanilla-JS frontend (`src/nexus_agent/webapp/`) lets you
submit a sample and watch it move through the real pipeline from a
browser, instead of invoking the graph from Python. It composes the
reviewer API (below) under the same origin at `/review/*`.

```bash
# 1. Local infra must be up first (see Quickstart) -- Postgres for run
#    tracking/corrections/escalations, MinIO for the uploaded image and
#    expression file.
docker compose up -d   # falls back to `docker-compose up -d` if your
                        # environment only has the v1 binary

# 2. Launch the app
conda activate nexus-agent
uvicorn nexus_agent.webapp.api:app --reload --host 127.0.0.1 --port 8420
```

Open `http://127.0.0.1:8420` in a browser. Running inside VS Code (locally
or over a remote/SSH/Codespaces connection)? Use its port forwarding
instead of a plain browser tab:

- VS Code usually auto-detects the listening port and offers an "Open in
  Browser" toast — click it, or
- open the **Ports** panel (**PORTS** tab next to Terminal, or
  `Ctrl+Shift+P` → "Ports: Focus on Ports View"), click **Forward a
  Port**, enter `8420`, then click the 🌐 icon on that row to open it.

A submitted run takes several minutes to reach `done` — real CellPose +
VGG16 inference on CPU, not a hang — the page polls in the background so
there's no need to keep it in the foreground.

The expression file isn't limited to `.h5ad`: `.loom`, a 10x Genomics
`.h5` feature-barcode matrix, a zipped/tarred 10x mtx bundle, or
`matrix.mtx` + `barcodes.tsv` + `features.tsv` (each optionally `.gz`)
selected together in the browser with no zipping step (the file picker's
multi-select) are all accepted and converted to the same `AnnData` shape
(`webapp/uploads.py::read_expression_upload`). Either the image or a
single expression file can also be plain-gzipped on its own (e.g.
`image.tiff.gz`, `expression.h5ad.gz`) -- it's decompressed automatically
before the usual extension-based dispatch. Seurat/SingleCellExperiment
`.rds` objects aren't read directly — R's native serialization needs a
real R+Bioconductor runtime to parse reliably, which this project doesn't
otherwise depend on. Convert on the R side first, e.g.
`SeuratDisk::SaveH5Seurat()` + `Convert(dest = "h5ad")`, or
`sceasy::convertFormat(obj, from = "seurat", to = "anndata")`, then upload
the resulting `.h5ad`. A 10x Genomics Loupe `.cloupe` file can't be read
at all -- unlike `.rds`, its format is proprietary and undocumented with
no reader anywhere outside 10x's own Loupe Browser to build on. Upload
the `*_feature_bc_matrix.h5` or mtx bundle it was generated from instead
(both already supported).

## Reviewer API (Phase 5)

Corrections and the human-escalation queue are served as a FastAPI app —
mounted at `/review/*` under the webapp above, or standalone:

```bash
uvicorn nexus_agent.review.api:app --reload
```

- `GET /escalations` / `POST /escalations/{id}/resolve` — the human-review
  queue Critic escalates low-confidence/contradictory results into.
- `POST /corrections` / `GET /corrections` — an expert correcting a
  cell-type label, spatial-domain call, or interpretation (Art. VII §1);
  these feed `learning/lora_finetune.py`'s fine-tune loop.
- `GET /metrics` — Prometheus exposition of veto rate, correction-rate
  trend, latency, and confidence distribution per agent (Art. XII §8).

Citations won't surface in a report until the starter literature corpus is
ingested once against a fresh database:

```python
from nexus_agent.xai.literature_rag import HashingLiteratureEmbedder, ingest_corpus
from nexus_agent.shared.config import settings
ingest_corpus(settings.postgres_dsn, HashingLiteratureEmbedder())
```

## Running a task

There's no CLI yet — the graph is invoked directly from Python. This runs
the full Coordinator → Vision → Analyst → Critic pipeline in-process, with
no persistence (use `compile_with_postgres` instead of `compile_with_memory`
to persist run state and inspect it afterward via `graph.get_state_history`):

```python
import uuid

from nexus_agent.graph.build import compile_with_memory
from nexus_agent.shared.schemas import AnalyticalGoal

graph = compile_with_memory()

run_id = uuid.uuid4()
initial_state = {
    "run_id": run_id,
    "task_id": uuid.uuid4(),
    "trace_id": uuid.uuid4(),
    # "modalities" drives Coordinator's subtask plan: both -> Vision then
    # Analyst; a single modality skips the agent that has nothing to do.
    "goal": AnalyticalGoal(sample_id="sample-1", modalities=["image", "expression"]),
    "history": [],
    "verdict": None,
    "force_verdict": None,
}

result = graph.invoke(initial_state, config={"configurable": {"thread_id": str(run_id)}})

print(result["verdict"])  # Verdict.PASS, .VETO, or .ESCALATE
for message in result["history"]:
    print(f"{message.from_agent} -> {message.to_agent} (confidence={message.confidence}): {message.payload}")
```

Every agent's contract is real (schema-validated `MessageEnvelope`s, a
Vision output matching Art. XII §2, Critic applying real confidence
thresholds), and as of Phase 3, Vision/Analyst run *real* model inference
too -- but only if you point the state at real data. Add `image_uri`/
`expression_uri` (object-store references, uploaded via
`nexus_agent.data.object_store.put_array`/`put_anndata`) to `initial_state`
and Vision will actually segment with CellPose and embed with VGG16, and
Analyst will actually fuse via the graph transformer, instead of emitting
the single-synthetic-cell stub shown above. See
[tests/test_graph_real_pipeline.py](tests/test_graph_real_pipeline.py) for
a complete example.

`AnalyticalGoal` also takes `run_spatial_analysis`/`run_enrichment_analysis`
(both default `False`) to opt into the Spatial Transcriptomics and Biology
(Enrichment) agents; both require `"expression"` in `modalities` and run
after Analyst, before Critic. QC runs automatically whenever `image_uri`/
`expression_uri` is set — no flag needed — and a `qc_verdict` of `"fail"`
routes the run straight to human review, skipping Vision/Analyst/Critic
entirely for that run.

On that real-pipeline path, Critic also generates Art. VI evidence (a
Grad-CAM++ heatmap, a SHAP gene-importance ranking once a Phase 5 checkpoint
has been promoted, and a RAG literature citation or an honest
"no supporting literature retrieved" flag) for every claim, persists it
independently of the report (`nexus_agent.xai.claims_evidence.
fetch_evidence_bundle`), and the graph's `report` node renders a
confidence-weighted HTML report into `result["report_html"]`. Cell typing
itself starts out `None` (Phase 3's placeholder) until at least one Phase 5
correction has gone through a fine-tuning cycle and been promoted — after
that, Analyst predicts from the latest promoted checkpoint automatically.

## Repo layout

```
src/nexus_agent/
├── shared/     # MessageEnvelope + AnalyticalGoal schemas, versioning, env config
├── bus/        # Redis Streams client — validates before publishing
├── data/       # SpatialData container, S3 object store client, provenance writer
├── agents/     # Coordinator, QC, Vision, Analyst, Spatial Transcriptomics,
│               # Biology (Enrichment), Critic — contracts + node logic
│               # (Art. III §1; Spatial/Biology/QC are Art. III §4 extensions)
├── vision/     # CellPose segmentation, VGG16/DINOv2 embedding (Art. XII §2)
├── analyst/    # Scanpy/Squidpy ingestion, graph-transformer fusion (Art. XII §3)
├── benchmark/  # Segmentation AJI metric + DSB2018 demo benchmark harness
├── xai/        # Grad-CAM++/attention rollout, SHAP, RAG literature layer,
│               # evidence store, explainability coverage metric (Art. VI)
├── report/     # Confidence-weighted report rendering (Art. VI §2)
├── review/     # FastAPI correction + escalation queue API (Art. VII §1, §3)
├── learning/   # Cell-type head, PEFT/LoRA fine-tune loop, promotion gate,
│               # feedback-value reporting (Art. VII §2/§4, Art. XII §6)
├── metrics/    # Veto-rate/latency/confidence dashboards (Art. XII §8)
├── graph/      # LangGraph StateGraph assembly and routing
└── webapp/     # Upload-and-view frontend: FastAPI app + static/ (HTML/CSS/JS,
                # no build step), mounts review/api.py at /review/*
tests/          # pytest; integration tests marked `@pytest.mark.integration`
db/schema.sql   # Postgres init script: provenance_log, correction_log,
                # literature_chunks, xai_evidence, message_log,
                # escalation_queue, finetune_runs
docker-compose.yml  # Postgres (pgvector/pgvector:pg16), Redis, MinIO
```

## Status

Phases 0-5 are implemented. Phases 0-3: a working agent graph —
Postgres-backed checkpointing, a validated message bus, real per-agent
contracts and procedural logic, and a real (if CPU-scoped — VGG16 default,
StarDist deferred; see [ROADMAP.md](ROADMAP.md)'s Phase 3 scope notes)
Vision/Analyst model stack producing 128-dim fused per-cell representations
with provenance intact. A real AJI benchmark (0.791 mean, DSB2018 demo
data) proves the validation harness, though full Art. VIII multi-platform
coverage remains open.

Phase 4: real Grad-CAM++, SHAP, and a pgvector-backed RAG literature layer
(against a small hand-authored starter corpus — not yet a real literature
database) generate Art. VI evidence for every claim, stored independently
of a confidence-weighted rendered report; explainability coverage measured
100% on a real end-to-end run. The ViT/attention-rollout path is wired but
not functional against the real DINOv2 checkpoint (disclosed in
[ROADMAP.md](ROADMAP.md) — its fused attention kernel never exposes
softmax weights to hook).

Phase 5: a real PEFT/LoRA fine-tuning loop, gated by a promotion regression
check, was verified end-to-end in this session — a correction submitted
through the reviewer API is genuinely reflected in a subsequent prediction
after a fine-tuning cycle, not just logged. See
[ROADMAP.md](ROADMAP.md)'s Phase 5 status note for a real bug this
verification found and fixed (the fusion transformer's base needed a fixed
seed to be reproducible across runs) and for the disclosed scope of the
LoRA implementation itself.
