-- NEXUS-Agent Phase 0 schema.
--
-- Deliberately does NOT define run_state: that table is owned and migrated
-- by LangGraph's PostgresSaver.setup() (see src/nexus_agent/graph/build.py).
-- This file only covers the two tables the Constitution (Art. XII §8) names
-- that LangGraph does not already manage. Structure only for Phase 0 —
-- populated starting Phase 4 (provenance_log) and Phase 5 (correction_log).

CREATE TABLE IF NOT EXISTS provenance_log (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL,
    task_id         UUID NOT NULL,
    claim_id        TEXT NOT NULL,
    source_image_region  JSONB,
    source_expression_profile JSONB,
    component       TEXT NOT NULL,
    component_version TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_provenance_log_run_id ON provenance_log (run_id);

CREATE TABLE IF NOT EXISTS correction_log (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL,
    claim_id        TEXT NOT NULL,
    reviewer        TEXT NOT NULL,
    original_value  JSONB NOT NULL,
    corrected_value JSONB NOT NULL,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_correction_log_run_id ON correction_log (run_id);
