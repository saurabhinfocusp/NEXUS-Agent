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
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Phase 5 (Art. XII §6): the fused per-cell representation at the time
    -- the claim was made, so a later fine-tune cycle can recover
    -- (cell_id -> fused_embedding) training pairs without a second
    -- object-store round trip. Nullable -- only real (non-stub) fused
    -- claims populate it.
    fused_embedding JSONB
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
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Phase 5 (Art. VII §1-2): which agent used to build the correction/
    -- fine-tune loop's training pairs and Art. VII §4's per-tissue-type
    -- feedback-value reporting. Nullable for backward compatibility with
    -- rows written before these columns existed.
    task_id         UUID,
    field           TEXT,
    tissue_type     TEXT
);

CREATE INDEX IF NOT EXISTS idx_correction_log_run_id ON correction_log (run_id);

-- Phase 4 (Art. VI §1.3, Art. XII §5): RAG literature layer's vector index.
-- 768 dims matches the named biomedical sentence-transformer's native
-- output (pritamdeka/S-PubMedBert-MS-MARCO) -- see xai/literature_rag.py.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS literature_chunks (
    id                      BIGSERIAL PRIMARY KEY,
    text                    TEXT NOT NULL,
    citation                TEXT NOT NULL,
    embedding               vector(768) NOT NULL,
    embedding_model_version TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Phase 4 (Art. VI §3): explainability artifacts stored as first-class
-- outputs, retrievable independently of the narrative report text.
CREATE TABLE IF NOT EXISTS xai_evidence (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              UUID NOT NULL,
    task_id             UUID NOT NULL,
    claim_id            TEXT NOT NULL,
    claim_type          TEXT NOT NULL,
    heatmap_uri         TEXT,
    shap_top_genes      JSONB,
    citations           JSONB,
    no_literature_retrieved BOOLEAN NOT NULL DEFAULT false,
    confidence          FLOAT NOT NULL,
    artifacts_expected  JSONB NOT NULL,
    artifacts_present   JSONB NOT NULL,
    component_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_xai_evidence_run_id ON xai_evidence (run_id);

-- Phase 4/5 (Art. XII §8): substrate for "measured, not log-mined"
-- dashboards -- veto rate, latency, confidence distribution per agent.
-- Populated once per emitted MessageEnvelope (see graph/build.py's
-- _with_message_logging wrapper), not scraped from checkpointer blobs.
CREATE TABLE IF NOT EXISTS message_log (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL,
    task_id     UUID NOT NULL,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    verdict     TEXT,
    confidence  FLOAT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_message_log_run_id ON message_log (run_id);
CREATE INDEX IF NOT EXISTS idx_message_log_task_id ON message_log (task_id);

-- Phase 5 (Art. VII §3): the human-review queue node from Phase 1/2 gets an
-- actual reviewer-facing surface (review/api.py) instead of a terminating
-- graph edge -- escalations are persisted here, not just logged.
CREATE TABLE IF NOT EXISTS escalation_queue (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL,
    task_id     UUID NOT NULL,
    claim_id    TEXT,
    payload     JSONB NOT NULL,
    confidence  FLOAT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_escalation_queue_status ON escalation_queue (status);

-- Phase 5 (Art. XII §6): fine-tune cycle + promotion-gate log.
CREATE TABLE IF NOT EXISTS finetune_runs (
    id                  BIGSERIAL PRIMARY KEY,
    triggered_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_corrections       INT NOT NULL,
    trigger_reason      TEXT NOT NULL,
    checkpoint_uri      TEXT,
    label_map           JSONB,
    benchmark_scores    JSONB,
    promoted            BOOLEAN NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- webapp (upload-and-view frontend): tracks an async pipeline run submitted
-- via POST /api/runs, since `graph.invoke()` (real CellPose/VGG16 inference)
-- takes minutes and must not block the HTTP request that started it.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          UUID PRIMARY KEY,
    task_id         UUID NOT NULL,
    sample_id       TEXT NOT NULL,
    image_uri       TEXT NOT NULL,
    expression_uri  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    verdict         TEXT,
    claims          JSONB,
    report_html     TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);
