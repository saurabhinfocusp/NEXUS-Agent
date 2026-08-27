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
  **Current status: Phase 0–3 closed, Phase 4 next.**

## Architecture at a glance

A LangGraph `StateGraph` routes work between four specialist agents —
Coordinator, Vision, Analyst, Critic/XAI — communicating via schema-validated
messages over a Redis Streams bus, with every claim in the final report
required to carry a visual attribution map, a feature-importance score,
and/or a literature citation (see Constitution Art. III and VI). Full
architectural and constitutional detail lives in the two docs linked above;
this README only covers running what's built so far.

## Quickstart

```bash
# 1. Create the conda environment (installs the project in editable mode,
#    including the Phase 3 model stack -- torch/torchvision from the CPU
#    wheel index, cellpose, scanpy, squidpy; ~2.5GB total including first-
#    run model weight downloads)
conda env create -f environment.yml
conda activate nexus-agent

# 2. Bring up local infra: Postgres, Redis, MinIO (S3-compatible storage)
cp .env.example .env   # adjust if needed
docker compose up -d

# 3. Run the tests
pytest tests/ -m "not integration" -q   # fast tests (no Docker, no model weights)
pytest tests/ -m integration            # integration tests: Docker + real
                                         # CellPose/VGG16 inference + a one-time
                                         # ~27MB dataset fetch for the AJI
                                         # benchmark (test_aji_benchmark.py
                                         # alone takes ~15-20 min on CPU)
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
a complete example. Real cell typing/domain assignment itself is still a
placeholder either way -- see [ROADMAP.md](ROADMAP.md)'s Phase 3 notes.

## Repo layout

```
src/nexus_agent/
├── shared/     # MessageEnvelope + AnalyticalGoal schemas, versioning, env config
├── bus/        # Redis Streams client — validates before publishing
├── data/       # SpatialData container, S3 object store client, provenance writer
├── agents/     # Coordinator, Vision, Analyst, Critic — contracts + node logic
├── vision/     # CellPose segmentation, VGG16/DINOv2 embedding (Art. XII §2)
├── analyst/    # Scanpy/Squidpy ingestion, graph-transformer fusion (Art. XII §3)
├── benchmark/  # Segmentation AJI metric + DSB2018 demo benchmark harness
└── graph/      # LangGraph StateGraph assembly and routing
tests/          # pytest; integration tests marked `@pytest.mark.integration`
db/schema.sql   # provenance_log / correction_log tables (Postgres init script)
docker-compose.yml  # Postgres, Redis, MinIO for local development
```

## Status

Phases 0-3 are complete: a working agent graph — Postgres-backed
checkpointing, a validated message bus, real per-agent contracts and
procedural logic (Coordinator's subtask planning, Critic's confidence
thresholds with a real re-route trigger), and a real (if CPU-scoped —
VGG16 default, StarDist deferred; see [ROADMAP.md](ROADMAP.md)'s Phase 3
scope notes) Vision/Analyst model stack: CellPose segmentation, VGG16/
DINOv2 embedding, Scanpy/Squidpy ingestion, and a graph-transformer fusion
module producing 128-dim per-cell representations with provenance intact.
A real AJI benchmark (0.791 mean, DSB2018 demo data) proves the validation
harness, though full Art. VIII multi-platform coverage remains open. Real
cell typing/domain assignment and explainability (Art. VI) land in
Phase 4+ per [ROADMAP.md](ROADMAP.md).
