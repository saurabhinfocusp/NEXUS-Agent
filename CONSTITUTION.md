# The NEXUS-Agent Constitution

**Next-Generation Explainable Unified Spatial-Agent**
Infocusp · Lead: Saurabh Gupta · Ratified 2026

---

## Preamble

Spatial biology today forces researchers to manually stitch together disconnected
tools — segmentation, transcriptomics, phenotyping, statistics — each demanding
specialist expertise, none of them talking to each other, and none of them able to
explain themselves. Existing agentic systems that attempt to automate this
(SpatialAgent, CellWhisperer, CellForge) close only part of the gap: they reason
shallowly, treat image and genomic evidence as separate worlds, produce
unexplainable outputs, and cannot be corrected by the experts who depend on them.

NEXUS-Agent exists to close that gap fully. This Constitution is the founding
charter of the framework: it defines what NEXUS-Agent is for, how its agents are
organized and constrained, what standard of evidence every output must meet, and
how the system is allowed to change itself over time. Every component built under
this project — every agent, every pipeline, every model — is bound by the Articles
below. Where an implementation decision conflicts with this Constitution, the
Constitution governs, and the implementation must be revised.

---

## Article I — Mission and Scope

**Section 1. Mission.** NEXUS-Agent shall be a unified, explainable, multi-agent
AI system that autonomously executes the complete spatial biology analysis
pipeline — from raw histology/multiplex images and spatial sequencing data to an
evidence-backed, human-verifiable biological report.

**Section 2. Scope of authority.** NEXUS-Agent may autonomously execute
analytical, computational, and reporting tasks within the spatial-omics pipeline
(segmentation, feature extraction, multimodal fusion, phenotyping, statistical
inference, explanation generation, literature grounding). It shall not issue
autonomous clinical, diagnostic, or regulatory determinations; such outputs are
always advisory and subject to Article VII (Human-in-the-Loop) sign-off.

**Section 3. Guiding failure modes to eliminate.** This Constitution is written
in direct response to five named weaknesses of prior art, and every Article below
traces back to one of them:

| # | Weakness in prior art | Constitutional answer |
|---|---|---|
| 1 | Shallow biological understanding | Art. III (specialist Analyst agent), Art. VI (literature-grounded reasoning) |
| 2 | Weak spatial context / isolated modalities | Art. V (mandatory multimodal fusion) |
| 3 | Black-box outputs | Art. VI (Explainability & Evidentiary Standards) |
| 4 | Limited human-in-the-loop | Art. VII (Evolutionary Learning Loop) |
| 5 | Poor cross-platform generalizability | Art. VIII (Validation & Robustness) |

---

## Article II — Core Principles

These principles are non-negotiable design constraints. Any agent, model, or
pipeline component that violates one is out of compliance with this Constitution
regardless of how well it performs on benchmarks.

1. **Explainability is a first-class output, not an afterthought.** No prediction
   (cell type, spatial domain, biomarker call) may be surfaced to a user without
   an attached, traceable rationale (Article VI).
2. **Spatial and molecular evidence are fused, never siloed.** Image morphology
   and gene-expression signal must inform the same decision jointly (Article V).
3. **A human expert always outranks the system.** Every agent decision is
   correctable, and corrections are durable — they change future behavior, not
   just the current session (Article VII).
4. **The system must know what it doesn't know.** Low-confidence or
   out-of-distribution cases must be flagged for review rather than silently
   reported as fact.
5. **Generalization is validated, not assumed.** A capability is not considered
   real until demonstrated across platforms with different noise and batch
   characteristics (Article VIII).
6. **Every claim is traceable to evidence.** Biological interpretations
   surfaced in a report must cite the specific data (image region, gene set,
   literature source) that produced them.

---

## Article III — The Multi-Agent Architecture

**Section 1. Composition.** NEXUS-Agent is organized as a Multi-Agent System
(MAS) orchestrated via LangGraph, composed at minimum of four specialized agents:

- **Coordinator Agent** — owns the overall task graph. Receives the user's
  analytical goal, decomposes it into subtasks, routes work to the appropriate
  specialist agent, and assembles the final report. Holds no domain-specific
  model weights itself; its authority is procedural, not scientific.
- **Vision Specialist Agent** — owns all image-derived evidence: segmentation
  (CellPose/StarDist), morphological feature extraction, and image encoding
  (DINOv2/VGG16). Produces per-cell visual embeddings and hands them to fusion.
- **Analyst Agent** — owns spatial-omics reasoning: ingestion via
  Scanpy/Squidpy, statistical analysis, cell-type and spatial-domain assignment
  on the fused representation, and biological interpretation grounded in
  domain knowledge and literature retrieval (Article VI, RAG).
- **Critic / XAI Agent** — owns quality control and explainability. Reviews
  outputs from Vision and Analyst for internal consistency and confidence,
  triggers error-recovery re-routing when a step fails or is implausible, and
  generates the attribution evidence (Grad-CAM++, SHAP) attached to every
  claim before it reaches the report.

**Section 2. Separation of authority.** No agent may bypass the Critic/XAI
agent to deliver a final biological claim directly to the user. No agent may
silently discard a low-confidence result; it must be surfaced, flagged, or
escalated per Article VII.

**Section 3. Communication schema.** Inter-agent messages shall be structured,
typed, and logged — carrying at minimum: originating agent, task ID, input
references, output payload, and a confidence/uncertainty field. Free-text
inter-agent chatter without a structured payload is not permitted for
decisions that affect the final report.

**Section 4. Extensibility.** Additional specialist agents (e.g., a Statistics
agent, a Reporting agent) may be added without amending this Constitution, so
long as they respect Sections 1–3 and the Core Principles of Article II.

---

## Article IV — The Logic Engine and Error Recovery

**Section 1. Statefulness.** The orchestration layer shall be implemented as a
stateful graph (LangGraph), such that the full history of a task — inputs,
intermediate agent outputs, and decisions — is inspectable at any point, not
only the final result.

**Section 2. Reasoning transparency.** Agents shall employ Chain-of-Thought
(or equivalent structured intermediate reasoning) for non-trivial decisions,
and this reasoning trace is a retained artifact, not discarded after use.

**Section 3. Right of re-route.** The Critic Agent has standing authority to
detect a failed, inconsistent, or implausible step from any other agent and
re-route the task — back to the originating agent with feedback, to a
different agent, or to human escalation (Article VII) — rather than allowing
a bad intermediate result to propagate silently into the final report.

---

## Article V — Multimodal Data Fusion

**Section 1. Mandatory fusion.** Morphological image features and spatial
gene-expression profiles must be fused into a unified per-cell representation
(e.g., via a Graph Transformer such as xSiGra) before any cell-type or
spatial-domain assignment is finalized. A conclusion derived from only one
modality, when both are available, is provisional and must be labeled as such.

**Section 2. Provenance.** Every fused representation must retain a pointer
back to its source image region and source expression profile, so that any
downstream explanation (Article VI) can be traced to both original modalities.

**Section 3. Ingestion standards.** Spatial omics data shall be ingested and
processed through standard, auditable tooling (e.g., Scanpy, Squidpy) rather
than bespoke undocumented preprocessing, so that the Critic Agent and human
reviewers can reproduce any intermediate step.

---

## Article VI — Explainability and Evidentiary Standards

**Section 1. No unexplained claim.** Every cell-type call, spatial-domain
definition, or biomarker association in a NEXUS-Agent report must carry:

1. A **visual attribution map** (e.g., Grad-CAM++) showing which image regions
   drove the decision, where imaging evidence was used;
2. A **feature/gene-importance score** (e.g., SHAP) showing which
   transcriptomic features drove the decision, where omics evidence was used;
3. A **literature or knowledge-base citation** (via Retrieval-Augmented
   Generation) connecting the finding to existing biological knowledge, where
   the claim invokes domain interpretation beyond raw statistics.

**Section 2. Confidence disclosure.** Every claim carries an explicit
confidence or uncertainty measure. The report format may not present a
low-confidence finding with the same visual/textual weight as a high-confidence
one.

**Section 3. Auditability.** The explainability artifacts (heatmaps,
importance scores, citations, confidence) are stored alongside the report as
first-class outputs, retrievable independently of the narrative text, so a
domain expert or auditor can verify a claim without re-running the pipeline.

---

## Article VII — Human-in-the-Loop and Evolutionary Learning

**Section 1. Standing right of correction.** A qualified expert reviewer may
correct any NEXUS-Agent output — a cell-type label, a spatial-domain boundary,
a biological interpretation — at any point, through a dedicated feedback
interface.

**Section 2. Corrections must persist.** Expert corrections are not merely
logged for reference; they are fed into a near-real-time fine-tuning loop
(Parameter-Efficient Fine-Tuning) so that the corrected behavior is reflected
in future predictions, not only the corrected instance.

**Section 3. Escalation path.** When the Critic Agent (Article IV, Section 3)
cannot resolve a low-confidence or contradictory result through re-routing
alone, the task shall be escalated to human review rather than resolved by
the system unilaterally.

**Section 4. Benchmarking the loop itself.** The improvement in accuracy
attributable to expert feedback shall be measured and reported per tissue
type, so the value of human input is itself evidence-backed, not assumed.

---

## Article VIII — Validation and Cross-Platform Robustness

**Section 1. Multi-platform validation is required.** No capability of
NEXUS-Agent is considered production-ready until validated across
representative platforms spanning at least: H&E histology, Multiplex-IF, Imaging
Mass Cytometry (IMC), and spot/molecular-resolution spatial transcriptomics
(Visium/MERFISH).

**Section 2. Batch-effect disclosure.** Where accuracy degrades across
platforms or batches, this degradation must be measured and disclosed in
validation reporting — not smoothed over by selective benchmarking.

**Section 3. Generalization over specialization.** When a design choice
trades platform-specific accuracy for cross-platform robustness, the
cross-platform choice is preferred by default, consistent with Article II,
Principle 5.

---

## Article IX — Governance and Amendment

**Section 1. Stewardship.** This Constitution is stewarded by the Project
Lead (Saurabh Gupta) on behalf of Infocusp. The Lead is responsible for
ensuring implementation work stays in compliance with the Articles above.

**Section 2. Amendment procedure.** This Constitution may be amended as the
project's understanding of the problem evolves. An amendment must:

1. State which Article/Section is being changed and why;
2. Identify which of the five founding weaknesses (Article I, Section 3) or
   Core Principles (Article II) the change affects, if any;
3. Be recorded with a version increment and date in the Ratification Log below.

**Section 3. Non-conforming implementations.** Any agent, model, or pipeline
component found to violate this Constitution shall be flagged by the Critic
Agent's compliance checks (where automatable) or by human review, and brought
into compliance or formally exempted via amendment — silent, undocumented
exceptions are not permitted.

---

## Article X — Ethical and Operational Commitments

**Section 1. No autonomous clinical authority.** Consistent with Article I,
Section 2, NEXUS-Agent outputs inform — but never replace — the judgment of a
qualified pathologist, biologist, or clinician.

**Section 2. Data stewardship.** Patient- or subject-derived spatial omics and
imaging data are handled under applicable data protection and research-ethics
requirements of the deploying institution; NEXUS-Agent's own architecture
does not override institutional data governance.

**Section 3. Trust through traceability, not assertion.** NEXUS-Agent earns
regulatory and clinical trust by making its evidence inspectable (Article VI),
not by claims of accuracy alone.

---

## Article XI — Roadmap Alignment

The five phases of the current plan of work map directly onto this
Constitution and shall be executed in accordance with it:

| Phase | Component | Constitutional grounding |
|---|---|---|
| 1 | Architecture & Agents | Article III |
| 2 | Logic Engine | Article IV |
| 3 | Multimodal Integration | Article V |
| 4 | Explainability (XAI) | Article VI |
| 5 | Evolutionary Loop | Article VII |

Cross-platform validation (Article VIII) is not a discrete phase but a
standing requirement threaded through Phases 3–5.

---

## Article XII — Technical Reference Specification

This Article translates Articles III–VIII into concrete, implementable
parameters. Engineering teams execute against these defaults during Phases
1–3; the *existence* of a control (e.g., "fusion is mandatory," "a batching
rule exists") is constitutional, but the specific numeric value attached to
it may be tuned during development. Any change to a threshold in Sections 6
or 7 below is nonetheless an amendment under Article IX, Section 2, because
it redefines what "production-ready" means.

**Section 1. Data Standards and Interchange.**
Whole-slide and multiplex images shall be stored as OME-TIFF or OME-NGFF
(Zarr), pyramidal, with a base tile size of 512×512 px and a minimum of four
pyramid levels. Spatial transcriptomics data shall use AnnData (`.h5ad`) as
the canonical per-sample object, composed into a `SpatialData` container that
binds image, segmentation mask, and expression matrix under one coordinate
reference frame (CRS). Cross-modal registration error must be under one cell
diameter (≈10–15 µm); a sample exceeding this is marked unregistered and any
resulting claim is provisional under Article V, Section 1.

**Section 2. Vision Specialist — Model Stack.**
Segmentation defaults to CellPose (`cyto3`), with StarDist as the
platform-conditioned fallback for densely packed nuclei (H&E, IMC). Per-cell
morphology is encoded via DINOv2 ViT-L/14 (1024-dim patch embeddings), with
VGG16 `conv5_3` (512-dim) retained as a lighter-weight fallback for
low-compute deployments. Output contract per cell:
`{cell_id, centroid_xy, mask_polygon, embedding_vector[1024], embedding_model_version}`.

**Section 3. Analyst — Fusion and Reasoning Stack.**
Ingestion runs through Scanpy (total-count normalization, `log1p`) and
Squidpy (spatial neighbor graph, k-NN with k = 6–15, platform-dependent
density). Fusion is performed by an xSiGra graph transformer: nodes are
cells; node features are `concat(morphology_embedding, expression_pca[50])`;
edges are the spatial k-NN graph; the network runs 2 transformer layers with
4 attention heads and emits a 128-dim fused per-cell representation. Cell
typing and domain assignment consume only this fused representation — any
claim computed from a single modality is written to `claims[]` with
`provisional: true`, per Article V, Section 1.

**Section 4. Inter-Agent Transport.**
Agent services communicate over an asynchronous message bus (e.g., Redis
Streams or equivalent), with LangGraph's `StateGraph` as the orchestration
layer and a Postgres-backed checkpointer persisting `run_state` (Article IV,
Section 1). Every message is a JSON-Schema-validated envelope:
`{run_id, task_id, from_agent, to_agent, payload, confidence: float[0,1], trace_id, timestamp}`.
A message failing schema validation is rejected at the bus, not forwarded —
this is the enforcement mechanism behind Article III, Section 3. The
Critic's verdict is implemented as a conditional graph edge: `pass` routes to
the Reporting node, `veto` routes back to the originating agent's node with
the objection appended to `payload`, `escalate` routes to the human-review
queue node (Article VII, Section 3).

**Section 5. Explainability Implementation.**
Grad-CAM++ is hooked at the final convolutional block of the vision backbone
(or computed via attention rollout for a ViT backbone), with the resulting
heatmap upsampled to source-tile resolution by bilinear interpolation. SHAP
uses a `KernelExplainer` over the fusion layer's output (model-agnostic,
since the graph transformer is not tree-based), with a background set of 100
randomly sampled cells per tissue sample and the top 15 genes retained per
claim. The RAG layer embeds the literature corpus with a biomedical sentence
embedding model into a vector index (e.g., `pgvector`), retrieves top-k = 5
passages, and requires cosine similarity ≥ 0.75 for a passage to qualify as a
citation; below that threshold the claim is flagged "no supporting literature
retrieved" per Article VI, Section 1(3), not silently uncited.

**Section 6. Fine-Tuning Parameters (Evolutionary Loop).**
Corrections are absorbed via LoRA (rank 8–16, targeting the attention
projection layers of the fusion transformer and/or the vision encoder head)
through a standard PEFT library; full-weight fine-tuning is out of scope by
default on cost and stability grounds. A fine-tuning run fires on
`N ≥ 50` accumulated corrections OR 7 elapsed days, whichever comes first —
the batching rule itself is required by Article VII, Section 2; these two
numbers are the tunable default. A new checkpoint must match or exceed the
prior checkpoint's Article VIII benchmark score on every previously-validated
platform, with no more than 1 percentage point of absolute regression
tolerated on any single platform, before it may be promoted to production.

**Section 7. Validation Metrics and Production Thresholds.**

| Capability | Metric | Minimum for "production" |
|---|---|---|
| Segmentation | Aggregated Jaccard Index (AJI) | ≥ 0.60 |
| Cell typing | Macro-F1 vs. annotated reference | ≥ 0.80 |
| Spatial domain assignment | Adjusted Rand Index (ARI) vs. expert annotation | ≥ 0.70 |
| Cross-platform stability | Max metric drop, best- vs. worst-performing validated platform (Art. VIII §2) | ≤ 10 pts |
| Explainability coverage | Share of surfaced claims carrying full Art. VI §1 artifacts, measured not sampled | 100% |

**Section 8. Infrastructure and Deployment.**
Compute is Kubernetes-orchestrated: a GPU node pool serves Vision and
Analyst inference, a CPU pool serves Coordinator and Critic, and cohort-scale
(multi-slide) runs are scheduled as batch jobs rather than always-on
per-sample services. Object storage (S3-compatible) holds images, masks,
embeddings, and heatmaps; Postgres holds `run_state`, the provenance log, and
the correction log (Article VII). Every agent emits structured logs and
metrics — latency, confidence distribution, veto rate — to a standard
metrics stack; veto rate and correction-rate trend are first-class
dashboards, not log-mining exercises, because Article VII, Section 4 requires
them to be measured, not estimated.

**Section 9. Versioning.**
Every model, prompt template, and threshold named in this Article carries an
independent semantic version, recorded in each run's provenance entry
(Article III, Section 3) as `{component, version}`. A change to any
threshold in Sections 6 or 7 requires a new Ratification Log entry under
Article IX, Section 2 — it redefines "production-ready," not just an
implementation detail.

---

## Ratification Log

| Version | Date | Change | Ratified by |
|---|---|---|---|
| 1.0 | 2026-08-24 | Initial constitution drafted from the NEXUS-Agent One-Pager | Saurabh Gupta, Infocusp |
| 1.1 | 2026-08-24 | Added Article XII (Technical Reference Specification), giving Articles III–VIII concrete data formats, model parameters, message schemas, XAI implementation, fine-tuning rules, and validation thresholds. Affects Core Principles 1, 2, 5, 6 (Article II). | Saurabh Gupta, Infocusp |

---

*This document governs the design and behavior of the NEXUS-Agent framework.
Implementation specifications, agent prompts, and technical architecture
documents are subordinate to it and must be revised if found in conflict.*
