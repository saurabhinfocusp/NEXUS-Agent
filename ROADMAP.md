# NEXUS-Agent Work Plan

**Status:** Phase 0–5 implemented (see per-phase status notes below) — derived from [CONSTITUTION.md](CONSTITUTION.md) v1.1
**Team assumption:** small (1–3 generalists), work is mostly sequential with
parallelization called out where it pays off once foundations exist.
**Sizing:** relative effort per task — `S` (days), `M` (~1–2 weeks solo),
`L` (~3+ weeks solo or a natural split-point for a second person).
No calendar dates; re-baseline sizes once the team and stack are fixed.

This plan operationalizes Article XI's five phases plus the Phase 0
scaffolding they all depend on. Article XII is the source of truth for every
concrete parameter below (model choices, schemas, thresholds) — where this
plan sets a number, it is repeating XII, not inventing one. **Any change to a
Section 6/7 threshold is a Constitutional amendment (Art. IX §2), not just an
edit to this file.**

---

## How to use this document

- Work top to bottom; a phase's exit criteria gate the next phase.
- Each phase lists **Constitutional grounding** so a task can always be
  traced back to the Article it satisfies (Art. II Principle 6 applies to
  this project's own process, not just its outputs).
- Checkboxes are implementation steps, not tickets — split further in your
  tracker of choice as needed.
- Update the "Status" line at the top and check off phases as they close.

---

## Phase 0 — Foundations & Scaffolding

*Not an Article XI phase itself, but everything in Phase 1+ needs it. Keep
this phase small — just enough to stop later phases from re-deciding
infrastructure mid-flight.*

**Constitutional grounding:** Art. XII §1 (data standards), §4 (transport),
§8 (infra) — read as prerequisites for Art. III–V.

### Steps

- [x] `S` Repo scaffolding: `src/nexus_agent/{shared,bus,data,graph}` +
      `tests/`. Coordinator/Vision/Analyst/Critic are stub *nodes* inside
      `graph/nodes.py` rather than separate top-level packages for now —
      there's no real per-agent logic yet to justify the split. Revisit the
      package boundary in Phase 1 once each agent gets real implementation
      (model calls, prompts) that warrants its own module.
- [x] `M` LangGraph `StateGraph` skeleton (`graph/build.py`, `graph/state.py`)
      with `PostgresSaver` checkpointing (`MemorySaver` swapped in only for
      fast unit tests) — proves Art. IV §1 (inspectable run state) from day
      one. Verified: `tests/test_graph_postgres.py` confirms a completed
      run's full history is queryable from the checkpointer afterward.
- [x] `M` `MessageEnvelope` pydantic model (`shared/schemas.py`) matching
      Art. XII §4 exactly: `{run_id, task_id, from_agent, to_agent, payload,
      confidence: float[0,1], trace_id, timestamp}`, `confidence` required
      and bounded. Bus validation wired in `bus/client.py::MessageBus.publish`
      — invalid messages raise `EnvelopeRejected` before `XADD`, never
      reaching the stream (Art. III §3 enforcement).
- [x] `S` Redis Streams bus (`bus/client.py`) — local docker-compose instance;
      production topology remains a Phase 5 concern.
- [x] `M` `SpatialData` container (`data/container.py`): binds image + mask +
      `AnnData` under one CRS, plus `check_registration()` (<1 cell diameter,
      15 µm threshold) marking a sample `unregistered` when it fails.
- [x] `S` Object storage convention: MinIO (S3-compatible) service in
      `docker-compose.yml` + connection settings in `shared/config.py` — no
      client wrapper yet, first real consumer is Phase 3+. Postgres tables
      for `provenance_log`/`correction_log` in `db/schema.sql`; `run_state`
      itself is owned by LangGraph's `PostgresSaver.setup()`, not hand-rolled.
- [x] `S` Versioning convention: `shared/versioning.py::stamp()` returns a
      registered `{component, version}` pair, stamped into every stub node's
      envelope payload.

### Exit criteria

- [x] A no-op task can flow: Coordinator → stub Vision → stub Analyst → stub
  Critic → done, with every hop logged as a schema-valid message and the
  full run inspectable from the checkpointer afterward. Also verified the
  `veto` and `escalate` conditional-edge paths (`tests/test_graph_smoke.py`)
  — veto correctly re-routes to the originating agent with the objection
  appended to `payload`, rather than terminating.
- [x] A synthetic sample loads into the `SpatialData` container end-to-end
  with `check_registration()` correctly passing/failing on aligned vs.
  offset coordinates (`tests/test_data_container.py`). Note: this used
  synthetic data, not a real registered slide — the "one real sample"
  version of this criterion is deferred to Phase 3 once real segmentation
  output exists to register against.

**Status: closed.** 19/19 tests passing (`pytest tests/`), Postgres/Redis/
MinIO running via `docker compose up -d`, initial commits on `main`. Ready
to start Phase 1.

---

## Phase 1 — Architecture & Agents

**Constitutional grounding:** Article III (in full).

### Steps

- [x] `M` Coordinator Agent (`agents/coordinator.py`): `AnalyticalGoal`
      (`sample_id`, `modalities`) in, `plan_subtasks()` decomposes it into
      an ordered `subtask_plan` — `[VISION, ANALYST]` / `[VISION]` /
      `[ANALYST]` depending on which modalities were requested. A lookup,
      no domain model weights, per Art. III §1.
- [x] `L` Vision Specialist Agent shell (`agents/vision.py`): output
      contract `VisionCellRecord` matches Art. XII §2 exactly
      (`cell_id, centroid_xy, mask_polygon, embedding_vector[1024],
      embedding_model_version`), pydantic-enforced (embedding vector must
      be exactly 1024-dim). Stub emits one synthetic record — real
      segmentation/encoding is still Phase 3.
- [x] `L` Analyst Agent shell (`agents/analyst.py`): `AnalystClaim`
      contract (`cell_id, cell_type, spatial_domain, confidence,
      provisional`). `provisional` is set from whether Vision ran for this
      task (Art. V §1) — the one piece of Analyst's contract the
      Constitution pins down ahead of Phase 3's real fusion/typing.
- [x] `M` Critic/XAI Agent shell (`agents/critic.py`): real default
      verdict logic — confidence < 0.15 → `escalate`, < 0.4 → `veto`, else
      `pass` — applied to the most recent reviewed message, not a hardcoded
      `pass`. The `force_verdict` test hook still exists but is now an
      override of real logic, not the only logic. XAI artifact generation
      (Grad-CAM++, SHAP) remains Phase 4.
- [x] `M` Veto/escalate contract as conditional graph edges (Art. XII §4)
      — carried over from Phase 0's `graph/build.py`, now driven by
      Critic's real logic above instead of a stub default.
- [x] `S` Art. III §2 enforced at the graph level — carried over from
      Phase 0 (no edge from Vision/Analyst to a terminal node except via
      Critic); still true with the new coordinator/vision conditional
      routing added this phase.
- [x] `S` Confidence field mandatory, non-defaultable — carried over from
      Phase 0's `MessageEnvelope` (`shared/schemas.py`).

### Parallelization note

Once the Coordinator + message contract exist, Vision-shell and
Analyst-shell can be built by two people in parallel — they only share the
schema, not each other's internals. With 1 person, do Coordinator → Critic
→ Vision → Analyst in that order (Critic before the specialists means the
veto contract exists before there's real output to veto, which surfaces
contract bugs early rather than at integration time).

### Exit criteria

- [x] All four agents exist as distinct graph nodes with typed, logged
  contracts; a synthetic (stub-model) run exercises `pass`, `veto`, and
  `escalate` paths at least once each (`tests/test_graph_smoke.py`,
  `tests/test_critic.py`).
- [x] No code path lets Vision or Analyst output reach a "final" state
  without passing through Critic — true for all three subtask-plan shapes
  (both modalities, image-only, expression-only), not just the full
  pipeline.

**Status: closed.** 35/35 tests passing (`pytest tests/`). Coordinator,
Vision, Analyst, Critic now live in `src/nexus_agent/agents/` with real
per-agent contracts and procedural logic; `AnalyticalGoal` lives in
`shared/schemas.py` rather than `agents/coordinator.py` to avoid a
coordinator↔state import cycle (`RunState` needs it as a real, non-deferred
import for LangGraph's schema introspection). Ready to start Phase 2.

---

## Phase 2 — Logic Engine and Error Recovery

**Constitutional grounding:** Article IV.

### Steps

- [x] `S` Full run history reconstructable from the Postgres checkpointer at
      an intermediate point, not just before/after — tested strictly
      between a veto and its retry
      (`tests/test_graph_postgres.py::test_run_history_is_reconstructable_at_a_point_strictly_between_veto_and_retry`).
- [x] `M` Structured `reasoning` field added to Coordinator's and Critic's
      payloads (`agents/coordinator.py`, `agents/critic.py`), persisted as
      part of the already-checkpointed `MessageEnvelope` (Art. IV §2).
- [x] `M` Re-route path hardened with a *real* trigger: `stub_confidence_override`
      (`graph/state.py`) lets a test make Vision/Analyst's first attempt
      genuinely low-confidence, so Critic's actual threshold logic (not the
      `force_verdict` hook) triggers the veto. `is_retry_after_veto`
      (`agents/common.py`) lets the re-routed agent detect it's retrying and
      report higher confidence -- acting on the objection, not just
      receiving it (`tests/test_error_recovery.py`).
- [x] `S` Escalation path terminates at `human_review` node, not a dead end
      — carried over from Phase 0/1, re-verified alongside the above.

### Exit criteria

- [x] A task history is fully reconstructable from the checkpointer at any
  intermediate point, including after a veto/re-route.
- [x] At least one integration test forces a bad intermediate result and
  verifies it never reaches the report un-flagged
  (`tests/test_error_recovery.py::test_genuinely_low_confidence_triggers_real_veto_and_never_reaches_report_unflagged`).

**Status: closed.**

---

## Phase 3 — Multimodal Data Fusion

**Constitutional grounding:** Article V; Art. XII §2–§3; threaded start of
Article VIII (cross-platform validation).

### Steps

- [x] `L` Vision model stack (`vision/segmentation.py`, `vision/embedding.py`):
      CellPose segmentation as default; DINOv2 ViT-L/14 wired but not the
      default (weight download is lazy/opt-in), VGG16 `conv5_3` is the
      *default* embedder on this CPU-only box, zero-padded 512→1024 to
      satisfy the fixed Art. XII §2 contract. **Scope reduction:** StarDist
      is deferred, not installed (TensorFlow-based; not worth the
      disk/dependency weight for a fallback path this phase's dataset
      doesn't exercise) — `stardist_segment()` raises `NotImplementedError`
      pointing here. **Disclosure:** CellPose 4.x itself deprecated the
      separate `cyto3` checkpoint in favor of one unified "Cellpose-SAM"
      model; `model_type="cyto3"` is accepted for spec-compatibility but
      resolves to that same unified (~1.15GB) checkpoint, not a
      standalone cyto3-specific one — noted in `segment_cells()`'s
      docstring rather than silently implied otherwise.
- [x] `M` Ingestion via Scanpy (total-count normalization, `log1p`) and
      Squidpy (spatial k-NN graph, k = 6–15, auto-capped for small inputs)
      in `analyst/ingestion.py`.
- [x] `L` Fusion transformer (`analyst/fusion.py`): node features =
      `concat(morphology_embedding[1024], expression_pca[50])`, spatial
      k-NN edges, 2 transformer layers, 4 attention heads, 128-dim fused
      output (Art. XII §3). **Scope reduction:** hand-rolled PyTorch
      (`nn.TransformerEncoder` with attention masked to the k-NN adjacency)
      instead of `torch_geometric` — see module docstring. Weights are
      randomly initialized (no training data/checkpoint exists yet): this
      proves the architecture is implemented correctly, not that fusion
      *quality* has been validated.
- [x] `M` Provenance pointers (`source_image_region`,
      `source_expression_profile` on every `FusedCellRecord`) written into
      the `provenance_log` table via `data/provenance.py`.
- [x] `S` Single-modality guardrail: `analyst_node` only sets
      `provisional=False` when Vision actually ran and fusion actually
      happened; single-modality claims are `provisional=True`
      (`tests/test_graph_real_pipeline.py::test_expression_only_stays_provisional`).
- [x] `M` Art. VIII validation tracking begun: `benchmark/aji.py` measures
      AJI against real CellPose predictions on StarDist's hosted DSB2018
      demo subset (fluorescence nuclei microscopy, no login). **Measured
      result** (3 images, 2026-08-27): mean AJI = **0.791** (0.726, 0.718,
      0.927) — comfortably above the Art. XII §7 ≥0.60 threshold. This is
      one real number on one platform, not Art. VIII §1's required
      multi-platform coverage (H&E, mIF, IMC, Visium/MERFISH) — DSB2018 is
      none of those. Full cross-platform validation remains open work for
      Phases 4-5.

### Exit criteria

- [x] End-to-end: raw image + expression data in (via `image_uri`/
  `expression_uri`, object-store references per Art. XII §8, not inline
  arrays) → fused 128-dim per-cell representation out, with provenance
  intact and single-modality claims correctly flagged provisional
  (`tests/test_graph_real_pipeline.py`).
- [x] First AJI measurement recorded (0.791 mean, DSB2018 demo subset) —
  real and honestly measured, but on a platform outside Art. VIII §1's
  four named ones; not yet "on a validated platform" in the full sense.

**Status: closed**, with the two scope reductions and the partial (not
full Art. VIII) validation coverage noted above carried forward as open
items for later phases rather than silently resolved.

---

## Phase 4 — Explainability (XAI)

**Constitutional grounding:** Article VI; Art. XII §5.

### Steps

- [x] `M` Grad-CAM++ hooked at the vision backbone's final conv block
      (`xai/gradcam.py::grad_cam_plusplus`, hooked at `Vgg16Embedder`'s
      conv5_3 — the CPU-only *default* embedder, Phase 3), upsampled to
      source-tile resolution via bilinear interpolation. Wired live into
      `agents/critic.py` for every claim with real Vision evidence.
      **Attention rollout (the ViT/DINOv2 path) is wired but not
      functional**: `xai/gradcam.py::attention_rollout` correctly raises a
      disclosed `RuntimeError` against the real `facebookresearch/dinov2`
      hub checkpoint, because that model's `Attention.forward` calls the
      fused `scaled_dot_product_attention` kernel directly and never
      materializes a separate post-softmax attention tensor to hook — there
      is nothing to roll out. Making this real would require monkey-patching
      DINOv2's attention forward pass to compute softmax eagerly; not done
      here. Since DINOv2 isn't the default embedder anyway (Phase 3), this
      doesn't block the default path, but it means the ViT branch of this
      bullet is honestly incomplete, not "done."
- [x] `M` SHAP via `KernelExplainer` over the fusion layer's output
      (`xai/shap_explain.py::shap_gene_importance`), 100 randomly sampled
      background cells per tissue sample (capped to all available cells
      when fewer), top 15 genes retained per claim. Wired live into
      `agents/critic.py`, gated on a Phase 5 promoted classifier existing
      (there's no real decision to explain before that).
- [x] `L` RAG literature layer (`xai/literature_rag.py`): default
      `HashingLiteratureEmbedder` (deterministic `sklearn.HashingVectorizer`,
      disclosed as a keyword-overlap stand-in, not real biomedical
      semantics) plus the named-spec `BioSentenceTransformerEmbedder`
      (`pritamdeka/S-PubMedBert-MS-MARCO`, lazy weight download, not
      default). Vector index is real `pgvector` (`literature_chunks`,
      `docker-compose.yml` now runs `pgvector/pgvector:pg16`), top-k = 5,
      cosine similarity ≥ 0.75 to qualify. Below threshold — or the corpus
      has nothing relevant — the claim is flagged
      `no_supporting_literature_retrieved` (Art. VI §1.3), never silently
      uncited. **Corpus is a small (35-entry) hand-authored starter set**
      (`xai/data/starter_corpus.json`) proving the mechanics end-to-end, not
      a real literature database — the "literature corpus source" open
      question below remains genuinely open.
- [x] `S` Confidence-weighted report rendering (`report/build.py`):
      `HIGH_CONFIDENCE_THRESHOLD`/`MEDIUM_CONFIDENCE_THRESHOLD` (new
      engineering values, same disclosure style as `critic.py`'s
      thresholds) drive a `weight_tier` whose visual weight (font-size,
      opacity, font-weight) is baked into `render_report_html`'s own
      per-tier styling, not exposed as a caller-settable parameter — a
      low-confidence entry structurally cannot render at the same weight as
      a high-confidence one (test-asserted, not just eyeballed).
- [x] `M` Auditability (`xai/claims_evidence.py`): heatmaps go to object
      storage, importance scores/citations/confidence/coverage flags go to
      the new `xai_evidence` table, all retrievable via
      `fetch_evidence_bundle(dsn, run_id, claim_id)` independent of report
      generation — verified end-to-end (real Postgres+pgvector, real
      pipeline run) that a claim's full evidence bundle is fetchable without
      touching `report/build.py` at all.
- [x] `S` Explainability coverage metric (`xai/coverage.py::
      explainability_coverage`) measures (queries `xai_evidence`, doesn't
      sample) the share of claims whose `artifacts_present` covers their
      `artifacts_expected`. Measured **1.0 on a real end-to-end run**
      (image+expression pipeline, real Grad-CAM++ heatmap + real RAG
      citation-or-flag; SHAP wasn't in `artifacts_expected` yet on that run
      since no checkpoint had been promoted, so it correctly wasn't counted
      against coverage).

### Exit criteria

- [x] Every claim type in a sample report carries all applicable artifacts
  from Art. VI §1, with confidence visibly differentiated — verified via a
  real end-to-end run against ephemeral Postgres+pgvector and mocked S3
  (Docker isn't available in this sandbox — see verification note below).
- [x] Explainability coverage metric read 100% (`1.0`) on that full run.

**Status: implemented, with the attention-rollout and starter-corpus scope
reductions above carried forward honestly, same as Phase 3's own disclosed
gaps.** 33 new fast tests pass (`pytest tests/ -q -m "not integration"`);
the Postgres/pgvector/object-store-dependent tests are real and correctly
written but skip in this sandbox (no Docker access — `docker compose up -d`
needs a `docker` group membership this session doesn't have). They were,
however, additionally verified for real in this session against an
ephemeral Postgres+pgvector (bundled `pgserver` binaries, TCP mode) and a
mocked S3 (`moto` server) — see the Phase 5 status note below for what that
run confirmed, since it exercised Phase 4 and Phase 5 together end-to-end.

---

## Phase 5 — Evolutionary Learning Loop

**Constitutional grounding:** Article VII; Art. XII §6.

### Steps

- [x] `L` Expert correction interface (`review/correction.py` +
      `review/api.py`): a FastAPI API layer (no rendered frontend — this
      repo has no UI code anywhere; confirmed as the right scope), lets a
      reviewer `POST /corrections` a `{field: cell_type|spatial_domain|
      interpretation, original_value, corrected_value, reviewer, reason,
      tissue_type}` correction. Writes straight into the Phase 0
      `correction_log` table (extended with `task_id`/`field`/
      `tissue_type` columns), not a separate event log.
- [x] `M` Escalation queue consumer (`review/escalation.py` +
      `review/api.py`'s `GET /escalations` / `POST /escalations/{id}/
      resolve`): `graph/build.py`'s `_human_review_node` now persists every
      escalation into a real `escalation_queue` table instead of being a
      terminating no-op edge (gated to the real pipeline so the fast
      stub-path tests never touch Postgres).
- [x] `L` PEFT/LoRA fine-tuning loop (`learning/lora_finetune.py`):
      rank-8 LoRA, through the real `peft` library, genuinely targets the
      fusion transformer's attention layers (`self_attn` as a whole —
      `nn.MultiheadAttention.forward` reads `out_proj` as a raw tensor for
      its fused kernel, never as a submodule call, so naming `out_proj`
      alone would silently produce an adapter that's never exercised;
      `peft`'s own `MultiheadAttention` docstring flags this — the module
      docstring explains this in full). Trigger rule
      (`should_trigger_finetune`): `N ≥ 50` accumulated corrections OR 7
      elapsed days, whichever first. **Disclosed scope reduction**: the
      actual persisted-correction training path (`run_finetune_cycle`)
      does *not* replay corrections through `GraphFusionTransformer` itself
      — only its already-fused 128-dim output is persisted per claim
      (`provenance_log.fused_embedding`), not the raw pre-fusion features —
      so it trains a small downstream `nn.Linear(128,128)` refiner +
      classifier directly (plain Adam, not PEFT) on those persisted pairs.
      The literal "LoRA on the fusion transformer's real attention layers"
      claim is demonstrated for real, separately, in a fast in-memory test
      (`test_lora_targets_real_fusion_transformer_attention`). Both facts
      are stated plainly in `learning/lora_finetune.py`'s docstring.
- [x] `M` Promotion gate (`learning/promotion.py::evaluate_promotion_gate`
      / `promote_checkpoint`): a new checkpoint is promoted only if every
      previously-validated platform's score is within 1 point of the prior
      checkpoint's — missing a previously-validated platform in the new
      scores counts as a failure, not a pass-by-omission.
- [x] `M` Feedback-value benchmarking (`learning/feedback_value.py::
      feedback_value_report`): per-`tissue_type`, `accuracy_before` is `0.0`
      by construction (a correction only exists where the system's original
      output was wrong), `accuracy_after` = the fraction of those same
      corrected cells the *latest promoted* checkpoint now predicts
      correctly — an honest, real, ongoing measure (Art. VII §4), not a
      fabricated global-accuracy number this repo has no held-out ground
      truth to compute.
- [x] `S` Metrics/dashboards (`metrics/dashboards.py`): `veto_rate`,
      `correction_rate_trend`, `latency_by_agent`, `confidence_distribution`
      all query the new `message_log` table (populated once per emitted
      envelope via `graph/build.py`'s `_with_message_logging` wrapper, gated
      to the real pipeline) — measured from a real store, not log-mined.
      `prometheus_exposition` is exposed at `GET /metrics` on the same
      FastAPI app.

### Exit criteria

- [x] **Verified end-to-end in this session, for real**: a correction
  submitted through `review/correction.py`'s interface *is* demonstrably
  reflected in a subsequent, independently-run `analyst_node` prediction
  after a fine-tuning cycle — not just logged. This required fixing one
  real bug discovered while verifying it (see status note below).
- [x] A checkpoint promotion was gated by the regression check with a real
  pass/fail outcome (`evaluate_promotion_gate`, unit-tested on both
  outcomes; exercised for real end-to-end too).

**Status: implemented and verified end-to-end, one real bug found and
fixed along the way.** Docker isn't reachable in this sandbox, so rather
than leave the correction → fine-tune → re-prediction loop merely
unit-tested-in-isolation, this session stood up an ephemeral Postgres (with
`pgvector`, via `pgserver`'s bundled binaries in TCP mode) and a mocked S3
(a local `moto` server) and ran the actual pipeline through it — both were
torn down afterward, nothing was left running.

That run surfaced a genuine architectural gap: `analyst/fusion.py::fuse()`
builds a **freshly random-initialized** `GraphFusionTransformer()` by
default on *every* call (Phase 3's own disclosed status — "no training
data/checkpoint exists yet"). That means the same cell got an unrelated
fused embedding on every pipeline run, so a correction trained against one
run's embedding could never actually improve a later run's prediction for
"the same" cell — only the training-time embedding, which
`feedback_value_report` measured as 100% correct while the live re-run
still failed. Fixed in `agents/analyst.py` (not Phase 3's
`analyst/fusion.py`, which stays untouched and closed) by seeding a fixed
`_fixed_seed_fusion_model()` base instead of the fully-random default —
still an untrained baseline exactly as Phase 3 disclosed, just now a fixed
(if arbitrary) function of its input rather than a new random one per call,
which is what a downstream classifier needs to hold onto a learned
association at all. `agents/critic.py`'s SHAP wiring uses the identical
fixed-seed base so its explanations stay in the same coordinate space the
promoted classifier was trained in. After the fix, the full loop — submit
correction → `run_finetune_cycle` → `promote_checkpoint` → fresh
`analyst_node` run — was re-verified to correctly predict every corrected
cell's corrected label, and `pytest tests/ -q` still shows 78 passed, 32
skipped, 0 failed (fast suite unaffected, no Postgres touched on the stub
path).

---

## Cross-cutting: Validation & Cross-Platform Robustness (Article VIII)

Not a phase — a standing requirement from Phase 3 onward. Track it as an
ongoing checklist rather than a milestone to "finish":

- [ ] Benchmark harness covers all four required platform types: H&E
      histology, Multiplex-IF, IMC, spot/molecular-resolution (Visium/MERFISH)
      (Art. VIII §1). **Not started** — Phase 3's `benchmark/aji.py` proves
      the harness mechanics on DSB2018 (fluorescence nuclei demo data),
      which is none of the four required types; a real platform-labeled
      dataset is still needed for each.
- [x] Segmentation AJI metric implementation exists and is measured for
      real (`aggregated_jaccard_index`, unit-tested against known
      pass/fail/partial-overlap cases; 0.791 mean on DSB2018, 3 images,
      2026-08-27) — the *metric* is validated even though platform
      coverage isn't yet.
- [ ] Per-platform, per-capability metrics tracked against Art. XII §7
      thresholds (AJI ≥0.60, cell-typing Macro-F1 ≥0.80, domain ARI ≥0.70,
      cross-platform max drop ≤10 pts) — only the AJI row has a real (if
      single-platform, non-required-type) number so far.
- [ ] Degradation across platforms/batches is disclosed in validation
      reporting, not smoothed over by selective benchmarking (Art. VIII §2).
- [ ] When a design choice trades platform-specific accuracy for
      cross-platform robustness, default to the cross-platform choice
      (Art. VIII §3, Art. II Principle 5) — log these tradeoff decisions
      somewhere reviewable, since they're easy to make silently.

"Production-ready" for any capability = every row in the Art. XII §7 table
satisfied simultaneously, not per-platform cherry-picking.

---

## Open questions to resolve before/during Phase 0

These aren't blocking a start, but resolve them early — they shape several
phases at once:

1. **Resolved (2026-08-27):** dev compute is CPU-only, no GPU. Art. XII §8's
   Kubernetes+GPU-pool target remains the eventual production goal; Phase 3
   defaulted to VGG16 over DINOv2 and skipped StarDist (TensorFlow) as a
   direct consequence — see Phase 3's scope-reduction notes above. Revisit
   defaults once real GPU compute is available.
2. **Partially resolved:** the Phase 4 RAG layer's mechanics (embedding,
   `pgvector` index, top-k + similarity-threshold retrieval, the
   "no supporting literature retrieved" flag) are real and proven against a
   small (35-entry) hand-authored starter corpus
   (`xai/data/starter_corpus.json`) — licensing and ingestion scope for a
   *real* literature corpus is still open.
3. **Partially resolved:** Phase 3's AJI harness is proven against DSB2018
   (fluorescence nuclei demo data, no login required) — 0.791 mean AJI,
   3 images. This is NOT one of Art. VIII §1's four required platform types
   (H&E, Multiplex-IF, IMC, Visium/MERFISH); picking and sourcing the first
   *required-type* labeled dataset is still open and needed before the
   cross-cutting Art. VIII checklist's first row can be checked off.

---

## Amendment discipline

If implementation reveals that a Section 6/7 number in Article XII is
wrong (e.g., the AJI threshold is unreachable given the chosen segmentation
model, or the LoRA rank needs to change), that's expected — Article XII says
so explicitly. Route it through Art. IX §2: state the Article/Section,
identify the affected Core Principle, and add a Ratification Log entry in
[CONSTITUTION.md](CONSTITUTION.md). Don't let this plan or the code silently
drift from a threshold the Constitution still states.
