"""QC Agent (Constitution Art. III §4: specialist agents beyond the
original four may be added without amending the Constitution, provided
they respect §§1-3).

Raw-data quality gate that runs before Vision/Analyst, not a biological-
claim producer -- so unlike Spatial/Biology it does **not** route through
Critic (mirrors Coordinator, which also isn't Critic-reviewed). Computes:

- Expression-side QC via Scanpy `sc.pp.calculate_qc_metrics` (total_counts,
  n_genes_by_counts, pct_counts_mt) -- this is what a real FASTQ->counts
  pipeline's own QC (STARsolo/Cell Ranger metrics) would normally feed;
  this repo never ingests raw FASTQ, only count matrices/.h5ad (see
  CLAUDE.md's tooling-scope note), so Scanpy's own QC metrics are the
  practical equivalent here.
- Image-side QC via scikit-image (already a dependency): Laplacian-variance
  focus/blur score plus an Otsu-threshold tissue-coverage fraction, as the
  practical equivalent of "QC visualization" for a histology tile.

When neither `image_uri` nor `expression_uri` is set, this falls back to
the Phase 0/1 stub convention shared by every other node: a fixed default
verdict, confidence resolved via `resolve_stub_confidence` so tests can
still exercise the fail/flag/pass routing without real data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from nexus_agent.agents.common import build_envelope, resolve_stub_confidence
from nexus_agent.graph.state import RunState
from nexus_agent.shared.schemas import AgentName
from nexus_agent.shared.versioning import stamp

# New Phase-N engineering values (Article XII does not name QC thresholds),
# same disclosure convention as agents/critic.py's VETO/ESCALATE thresholds.
QC_MIN_TOTAL_COUNTS = 200.0
QC_MAX_MT_PCT = 20.0
QC_MIN_FOCUS_SCORE = 1e-3

QC_FAIL_CONFIDENCE_THRESHOLD = 0.3
QC_FLAG_CONFIDENCE_THRESHOLD = 0.6


class QCReport(BaseModel):
    """QC agent's output contract: raw metrics plus the bucketed verdict
    the new `_qc_router` (graph/build.py) reads.
    """

    total_counts_mean: float | None = None
    n_genes_by_counts_mean: float | None = None
    pct_counts_mt_mean: float | None = None
    image_focus_score: float | None = None
    image_tissue_coverage_fraction: float | None = None
    qc_plot_uri: str | None = None
    qc_verdict: Literal["pass", "flag", "fail"]


def _verdict_for(confidence: float) -> Literal["pass", "flag", "fail"]:
    if confidence < QC_FAIL_CONFIDENCE_THRESHOLD:
        return "fail"
    if confidence < QC_FLAG_CONFIDENCE_THRESHOLD:
        return "flag"
    return "pass"


def _render_qc_plot(adata) -> bytes:
    """A matplotlib violin/histogram QC plot -- the practical equivalent of
    "QC visualization" -- pushed to object storage the same way Critic
    pushes Grad-CAM++ heatmaps (`xai/claims_evidence.py`), so it's a
    first-class artifact rather than a log line.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(adata.obs["total_counts"], bins=20)
    axes[0].set_title("total_counts")
    axes[1].hist(adata.obs["pct_counts_mt"], bins=20)
    axes[1].set_title("pct_counts_mt")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _expression_qc(expression_uri: str):
    """Real expression-side QC via Scanpy. Returns (metrics dict, quality
    score in [0, 1], the QC-annotated AnnData for plotting).
    """
    import scanpy as sc

    from nexus_agent.data.object_store import get_anndata

    adata = get_anndata(expression_uri)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    total_counts_mean = float(adata.obs["total_counts"].mean())
    n_genes_mean = float(adata.obs["n_genes_by_counts"].mean())
    pct_mt_mean = float(adata.obs["pct_counts_mt"].mean())

    score = 1.0 - min(pct_mt_mean / QC_MAX_MT_PCT, 1.0)
    if total_counts_mean < QC_MIN_TOTAL_COUNTS:
        score *= 0.5
    score = max(0.0, min(1.0, score))

    metrics = {
        "total_counts_mean": total_counts_mean,
        "n_genes_by_counts_mean": n_genes_mean,
        "pct_counts_mt_mean": pct_mt_mean,
    }
    return metrics, score, adata


def _image_qc(image_uri: str):
    """Real image-side QC via scikit-image. Returns (metrics dict, quality
    score in [0, 1]).
    """
    from skimage.color import rgb2gray
    from skimage.filters import laplace, threshold_otsu

    from nexus_agent.data.object_store import get_array

    image = get_array(image_uri)
    gray = rgb2gray(image) if image.ndim == 3 else image.astype(float)
    focus_score = float(laplace(gray).var())
    try:
        threshold = threshold_otsu(gray)
        coverage = float((gray > threshold).mean())
    except ValueError:
        # Otsu needs at least two distinct intensities; a flat/blank image
        # has zero tissue coverage by construction.
        coverage = 0.0

    score = 1.0 if focus_score >= QC_MIN_FOCUS_SCORE else focus_score / QC_MIN_FOCUS_SCORE
    score = max(0.0, min(1.0, score))

    metrics = {"image_focus_score": focus_score, "image_tissue_coverage_fraction": coverage}
    return metrics, score


def qc_node(state: RunState) -> dict:
    plan = state["subtask_plan"]
    next_specialist = (
        AgentName.VISION if AgentName.VISION in plan else (AgentName.ANALYST if AgentName.ANALYST in plan else AgentName.CRITIC)
    )

    image_uri = state.get("image_uri")
    expression_uri = state.get("expression_uri")

    report_kwargs: dict = {}
    scores: list[float] = []
    reasoning: list[str] = []
    qc_plot_uri: str | None = None

    if expression_uri:
        expr_metrics, expr_score, adata = _expression_qc(expression_uri)
        report_kwargs.update(expr_metrics)
        scores.append(expr_score)
        reasoning.append(
            f"expression QC via Scanpy calculate_qc_metrics on {expression_uri} -> score={expr_score:.2f}"
        )
        try:
            from nexus_agent.data.object_store import put_bytes

            qc_plot_uri = put_bytes(
                f"qc/{state['run_id']}/{state['task_id']}/qc_plot.png", _render_qc_plot(adata)
            )
        except Exception:
            qc_plot_uri = None  # best-effort artifact; never blocks the QC verdict

    if image_uri:
        image_metrics, image_score = _image_qc(image_uri)
        report_kwargs.update(image_metrics)
        scores.append(image_score)
        reasoning.append(
            f"image QC via Laplacian-variance focus + Otsu tissue coverage on {image_uri} -> score={image_score:.2f}"
        )

    if scores:
        confidence = min(scores)
    else:
        confidence, stub_reasoning = resolve_stub_confidence(state, AgentName.QC, default=0.9, retry_confidence=0.9)
        reasoning.extend(stub_reasoning)

    verdict = _verdict_for(confidence)
    report = QCReport(qc_verdict=verdict, qc_plot_uri=qc_plot_uri, **report_kwargs)
    # `fail` escalates straight to human_review (Art. III §2 doesn't bind QC
    # -- it's a raw-data gate, not a biological claim), matching what
    # `_qc_router` (graph/build.py) actually does with this verdict.
    to_agent = AgentName.HUMAN_REVIEW if verdict == "fail" else next_specialist

    envelope = build_envelope(
        state,
        from_agent=AgentName.QC,
        to_agent=to_agent,
        payload={
            "component_version": stamp(AgentName.QC).model_dump(mode="json"),
            "qc_report": report.model_dump(mode="json"),
            "reasoning": reasoning,
        },
        confidence=confidence,
    )
    return {"history": [envelope], "qc_verdict": verdict}
