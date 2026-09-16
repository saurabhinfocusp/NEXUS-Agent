# AGENTS.md

This repository is NEXUS-Agent, a multi-agent spatial biology analysis pipeline that turns histology and expression data into an evidence-backed biological report.

## Primary references

- [README.md](README.md) — quickstart, running the app, and task invocation
- [CONSTITUTION.md](CONSTITUTION.md) — binding design spec for graph behavior and evidence standards
- [ROADMAP.md](ROADMAP.md) — phase status and what is implemented vs. intentionally deferred
- [CLAUDE.md](CLAUDE.md) — repo-specific operational notes for local iteration

## Critical project conventions

- The graph is the source of truth for control flow: see [src/nexus_agent/graph/build.py](src/nexus_agent/graph/build.py) and [src/nexus_agent/graph/state.py](src/nexus_agent/graph/state.py).
- Inter-agent communication happens via validated message objects stored in state history; do not assume the Redis bus is the runtime path for graph execution.
- The synthetic stub path is intentionally no-op when there are no image/expression URIs on state. Preserve that separation when changing a node or wrapper that touches Postgres or object storage.
- LangGraph checkpoint tables are managed by PostgresSaver; do not add them to [db/schema.sql](db/schema.sql).
- There is no standalone CLI. Run the graph directly from Python or use the FastAPI webapp surface.

## Commands to use

```bash
# Fast unit tests
pytest tests/ -m "not integration" -q

# Integration tests (Docker, Postgres/pgvector, real model runs)
pytest tests/ -m integration

# App
uvicorn nexus_agent.webapp.api:app --reload --host 127.0.0.1 --port 8420

# Reviewer API
uvicorn nexus_agent.review.api:app --reload
```

## Working style for agents

- Prefer the narrowest relevant test when validating a change.
- Before changing routing or critic logic, read the graph router and the relevant tests, not just the agent implementation.
- Keep "real path" and "synthetic stub path" behavior distinct. The stub path is the fast test path and must remain in-memory.
- Use the existing config pattern from [src/nexus_agent/shared/config.py](src/nexus_agent/shared/config.py); do not invent a new dependency injection layer.
- When fixing webapp or upload logic, read the specific handlers in [src/nexus_agent/webapp](src/nexus_agent/webapp) and the matching tests in [tests](tests).

## Safety and scope notes

- Do not add new R-based or FASTQ-based pipeline paths; this codebase intentionally avoids direct R/FASTQ interop and expects Python-native processing for QC and spatial analysis.
- Keep outputs evidence-backed and schema-validated; if a new agent or message is introduced, update the shared schema and tests accordingly.
- When a behavior looks odd, check the Constitution before changing it.
