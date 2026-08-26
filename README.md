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
  **Current status: Phase 0 closed, Phase 1 next.**

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

## Repo layout

```
src/nexus_agent/
├── shared/     # MessageEnvelope schema, component versioning, env config
├── bus/        # Redis Streams client — validates before publishing
├── data/       # SpatialData container + cross-modal registration check
└── graph/      # LangGraph StateGraph: Coordinator -> Vision -> Analyst -> Critic
tests/          # pytest; integration tests marked `@pytest.mark.integration`
db/schema.sql   # provenance_log / correction_log tables (Postgres init script)
docker-compose.yml  # Postgres, Redis, MinIO for local development
```

## Status

Phase 0 (foundations & scaffolding) is complete: a working agent graph with
Postgres-backed checkpointing, a validated message bus, and a data container
with a registration-error check — all stubs for now, with real model/domain
logic landing per the phases in [ROADMAP.md](ROADMAP.md).
