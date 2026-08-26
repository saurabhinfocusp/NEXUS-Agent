# NEXUS-Agent Work Plan

**Status:** Phase 0 closed, Phase 1 next — derived from [CONSTITUTION.md](CONSTITUTION.md) v1.1
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

- [ ] `M` Coordinator Agent: task decomposition from a user analytical goal
      into a subtask graph, routing to specialist agents. No domain model
      weights here — keep its logic strictly procedural per Art. III §1.
- [ ] `L` Vision Specialist Agent shell: owns segmentation and encoding
      calls (implementation deferred to Phase 3's model stack), but the
      *agent* — its input/output contract, its place in the graph — is
      built now. Output contract per cell must match Art. XII §2:
      `{cell_id, centroid_xy, mask_polygon, embedding_vector[1024],
      embedding_model_version}`.
- [ ] `L` Analyst Agent shell: owns ingestion, statistical analysis, typing
      and domain assignment, biological interpretation. Model/fusion logic
      lands in Phase 3; the agent boundary and contract land now.
- [ ] `M` Critic/XAI Agent shell: consumes Vision + Analyst outputs, checks
      internal consistency and confidence, and can emit `pass` / `veto` /
      `escalate`. XAI artifact generation (Grad-CAM++, SHAP) is Phase 4 —
      here, just the control logic and verdict contract.
- [ ] `M` Wire the veto/escalate contract as conditional graph edges
      (Art. XII §4): `pass` → Reporting node, `veto` → back to originating
      agent's node with objection appended to `payload`, `escalate` →
      human-review queue node. This is the mechanism behind Art. IV §3
      (right of re-route) — build it structurally, not as an if/else buried
      in a handler.
- [ ] `S` Enforce Art. III §2 at the graph level: no edge exists from Vision
      or Analyst directly to the final report/user-facing node — every path
      to "delivered" routes through Critic.
- [ ] `S` Confidence field is mandatory, non-defaultable in every agent's
      output payload (Art. III §3) — validate at the schema layer from
      Phase 0, not by convention.

### Parallelization note

Once the Coordinator + message contract exist, Vision-shell and
Analyst-shell can be built by two people in parallel — they only share the
schema, not each other's internals. With 1 person, do Coordinator → Critic
→ Vision → Analyst in that order (Critic before the specialists means the
veto contract exists before there's real output to veto, which surfaces
contract bugs early rather than at integration time).

### Exit criteria

- All four agents exist as distinct graph nodes with typed, logged
  contracts; a synthetic (stub-model) run exercises `pass`, `veto`, and
  `escalate` paths at least once each.
- No code path lets Vision or Analyst output reach a "final" state without
  passing through Critic.

---

## Phase 2 — Logic Engine and Error Recovery

**Constitutional grounding:** Article IV.

### Steps

- [ ] `S` Confirm full run history (inputs, intermediate outputs, decisions)
      is queryable from the Postgres checkpointer, not just the final
      result — this was scaffolded in Phase 0; here it gets tested against
      a multi-step real run, including after a veto/re-route cycle.
- [ ] `M` Add a structured reasoning trace to each agent's non-trivial
      decisions (Chain-of-Thought or equivalent) and persist it as a
      retained artifact (Art. IV §2) — not just logged for debugging, but
      stored where Phase 4's auditability requirements can retrieve it.
- [ ] `M` Exercise and harden the re-route path end to end: inject a
      deliberately failing/implausible intermediate result and confirm
      Critic catches it, re-routes with feedback, and the originating agent
      can act on that feedback (not just receive it).
- [ ] `S` Escalation stub: when Critic can't resolve via re-routing, task
      lands in a human-review queue (full review *interface* is Phase 5;
      here, just prove the escalate edge terminates somewhere real, not a
      dead end).

### Exit criteria

- A task history is fully reconstructable from the checkpointer at any
  intermediate point, including after a veto/re-route.
- At least one integration test forces a bad intermediate result and
  verifies it never reaches the report un-flagged.

---

## Phase 3 — Multimodal Data Fusion

**Constitutional grounding:** Article V; Art. XII §2–§3; threaded start of
Article VIII (cross-platform validation).

### Steps

- [ ] `L` Vision model stack (Art. XII §2): CellPose (`cyto3`) segmentation
      as default, StarDist fallback for densely packed nuclei (H&E, IMC).
      DINOv2 ViT-L/14 for per-cell morphology embeddings (1024-dim), VGG16
      `conv5_3` as the lighter-weight fallback path.
- [ ] `M` Ingestion via Scanpy (total-count normalization, `log1p`) and
      Squidpy (spatial k-NN graph, k = 6–15, platform-dependent) — "standard,
      auditable tooling," per Art. V §3, not bespoke preprocessing.
- [ ] `L` xSiGra graph transformer fusion: nodes = cells, node features =
      `concat(morphology_embedding, expression_pca[50])`, edges = spatial
      k-NN graph, 2 transformer layers, 4 attention heads, 128-dim fused
      output (Art. XII §3).
- [ ] `M` Provenance pointers: every fused representation retains a pointer
      back to its source image region and source expression profile
      (Art. V §2) — this is what makes Phase 4's attribution traceable, so
      don't defer it.
- [ ] `S` Single-modality guardrail: cell-typing/domain-assignment logic
      must only consume the fused representation. Any code path that would
      compute a claim from one modality alone must tag it
      `claims[].provisional = true` (Art. V §1, Art. XII §3) — make this
      structurally hard to skip, e.g. the typing function simply doesn't
      accept an unfused input type.
- [ ] `M` Begin Art. VIII validation tracking here (it's threaded through
      Phases 3–5, not a separate phase): stand up the benchmark harness and
      start recording Segmentation AJI on at least one platform, so the
      ≥0.60 threshold (Art. XII §7) has a number attached before Phase 4
      claims completeness.

### Exit criteria

- End-to-end: raw image + expression data in → fused 128-dim per-cell
  representation out, with provenance intact and single-modality claims
  correctly flagged provisional.
- First AJI measurement recorded on at least one validated platform.

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
      (Art. VIII §1).
- [ ] Per-platform, per-capability metrics tracked against Art. XII §7
      thresholds (AJI ≥0.60, cell-typing Macro-F1 ≥0.80, domain ARI ≥0.70,
      cross-platform max drop ≤10 pts).
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

1. **Compute target for Phase 0–3**: Art. XII §8 assumes Kubernetes with a
   GPU pool; confirm whether that's available now or if early phases should
   develop against a single-GPU dev box with K8s deferred to Phase 5's infra
   hardening.
2. **Literature corpus source** for the Phase 4 RAG layer — licensing and
   ingestion scope affects sizing of that task significantly.
3. **First validation platform** for Phase 3's initial AJI measurement —
   picking the platform with the most accessible annotated reference data
   first will unblock Article VIII tracking soonest.

---

## Amendment discipline

If implementation reveals that a Section 6/7 number in Article XII is
wrong (e.g., the AJI threshold is unreachable given the chosen segmentation
model, or the LoRA rank needs to change), that's expected — Article XII says
so explicitly. Route it through Art. IX §2: state the Article/Section,
identify the affected Core Principle, and add a Ratification Log entry in
[CONSTITUTION.md](CONSTITUTION.md). Don't let this plan or the code silently
drift from a threshold the Constitution still states.
