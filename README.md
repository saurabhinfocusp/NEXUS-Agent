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
  **Current status: Phase 0 & 1 closed, Phase 2 next.**

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
# 1. Create the conda environment (installs the project in editable mode)
conda env create -f environment.yml
conda activate nexus-agent

# 2. Bring up local infra: Postgres, Redis, MinIO (S3-compatible storage)
cp .env.example .env   # adjust if needed
docker compose up -d

# 3. Run the tests
pytest tests/ -q                      # fast tests (no Docker required)
pytest tests/test_graph_postgres.py   # integration test, needs Postgres up
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

As of Phase 1, every agent's *contract* is real (schema-validated
`MessageEnvelope`s, a Vision output matching Art. XII §2, Critic applying
real confidence thresholds) but the actual model inference inside each
agent — segmentation, fusion, cell typing — is still a stub; that lands in
Phase 3 onward per [ROADMAP.md](ROADMAP.md).

## Repo layout

```
src/nexus_agent/
├── shared/     # MessageEnvelope + AnalyticalGoal schemas, versioning, env config
├── bus/        # Redis Streams client — validates before publishing
├── data/       # SpatialData container + cross-modal registration check
├── agents/     # Coordinator, Vision, Analyst, Critic — contracts + node logic
└── graph/      # LangGraph StateGraph assembly and routing
tests/          # pytest; integration tests marked `@pytest.mark.integration`
db/schema.sql   # provenance_log / correction_log tables (Postgres init script)
docker-compose.yml  # Postgres, Redis, MinIO for local development
```

## Status

Phase 0 (foundations & scaffolding) and Phase 1 (agent architecture) are
complete: a working agent graph — Postgres-backed checkpointing, a
validated message bus, a data container with a registration-error check,
and real per-agent contracts/routing logic (Coordinator's subtask
planning, Critic's confidence-threshold verdicts) — with actual model
inference still stubbed out pending the phases in [ROADMAP.md](ROADMAP.md).
