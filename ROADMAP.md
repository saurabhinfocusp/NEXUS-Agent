# NEXUS-Agent Work Plan

**Status:** Phase 0–3 closed, Phase 4 next — derived from [CONSTITUTION.md](CONSTITUTION.md) v1.1
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

- [ ] `M` Grad-CAM++ hooked at the vision backbone's final conv block (or
      attention rollout if the backbone is a ViT), upsampled to source-tile
      resolution via bilinear interpolation.
- [ ] `M` SHAP via `KernelExplainer` over the fusion layer's output
      (model-agnostic — the graph transformer isn't tree-based), 100
      randomly sampled background cells per tissue sample, top 15 genes
      retained per claim.
- [ ] `L` RAG literature layer: biomedical sentence-embedding model over the
      literature corpus into a vector index (`pgvector` or equivalent),
      top-k = 5 retrieval, cosine similarity ≥ 0.75 to qualify as a
      citation. Below threshold, the claim is flagged "no supporting
      literature retrieved" (Art. VI §1.3) — never silently uncited.
- [ ] `S` Confidence-weighted report rendering: a low-confidence finding
      cannot share the same visual/textual weight as a high-confidence one
      (Art. VI §2) — this is a report-template constraint, enforce it in
      the rendering layer, not by author discipline.
- [ ] `M` Auditability: store heatmaps, importance scores, citations, and
      confidence as first-class outputs retrievable independently of the
      narrative text (Art. VI §3) — verify by pulling a claim's full
      evidence bundle without touching the report generator.
- [ ] `S` Wire the "Explainability coverage" metric from Art. XII §7:
      measure (not sample) the share of surfaced claims carrying full
      Art. VI §1 artifacts — this needs to hit 100% before anything is
      called production-ready.

### Exit criteria

- Every claim type (cell-type call, spatial-domain definition, biomarker
  association) in a sample report carries all applicable artifacts from
  Art. VI §1, with confidence visibly differentiated.
- Explainability coverage metric reads 100% on at least one full run.

---

## Phase 5 — Evolutionary Learning Loop

**Constitutional grounding:** Article VII; Art. XII §6.

### Steps

- [ ] `L` Expert correction interface: lets a qualified reviewer correct a
      cell-type label, domain boundary, or interpretation (Art. VII §1).
      Corrections write to the Phase 0 correction log, not just a UI event.
- [ ] `M` Escalation queue UI/consumer: the human-review queue node from
      Phase 1/2 needs an actual reviewer-facing surface now, not just a
      terminating graph edge.
- [ ] `L` PEFT/LoRA fine-tuning loop (Art. XII §6): rank 8–16, targeting
      attention projection layers of the fusion transformer and/or vision
      encoder head. Trigger rule: fires on `N ≥ 50` accumulated corrections
      OR 7 elapsed days, whichever first — both numbers are tunable
      defaults, but the batching rule's *existence* is required.
- [ ] `M` Promotion gate: a new checkpoint must match or exceed the prior
      checkpoint's Art. VIII benchmark score on every previously-validated
      platform, ≤1 point absolute regression tolerated on any single
      platform, before production promotion (Art. XII §6).
- [ ] `M` Feedback-value benchmarking (Art. VII §4): measure accuracy
      improvement attributable to expert feedback, reported per tissue
      type — this is a dashboard/report, not a one-off analysis, since the
      Constitution wants the value of human input evidence-backed on an
      ongoing basis.
- [ ] `S` Metrics/dashboards from Art. XII §8: veto rate and correction-rate
      trend as first-class dashboards (not log-mining), latency and
      confidence distribution per agent.

### Exit criteria

- A correction submitted through the interface is demonstrably reflected in
  a subsequent prediction after a fine-tuning cycle (not just logged).
- A checkpoint promotion has been gated at least once by the regression
  check, with a real pass/fail outcome.

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
2. **Literature corpus source** for the Phase 4 RAG layer — licensing and
   ingestion scope affects sizing of that task significantly. Still open.
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
