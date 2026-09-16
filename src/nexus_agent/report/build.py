"""Report assembly and rendering (Constitution Art. VI -- Explainability and
Evidentiary Standards).

This module owns two things:

1. `render_report` -- assembling a structured `Report` (one `ClaimReportEntry`
   per claim) from a plain, decoupled `ReportClaimInput` list plus an
   evidence-by-claim-id lookup. It deliberately does not import
   `agents/analyst.py`'s `AnalystClaim` or `xai/claims_evidence.py`'s
   `ClaimEvidenceBundle` -- both are owned elsewhere and may be built/changed
   concurrently. A caller (expected to eventually live in `agents/critic.py`)
   is responsible for translating those richer types into the plain shapes
   this module accepts: a `ReportClaimInput` per claim, and a plain
   `{"heatmap_uri":..., "shap_top_genes":..., "citations":...,
   "no_literature_retrieved":...}` dict per claim_id.
2. `render_report_html` -- rendering a `Report` to a self-contained HTML
   document. Art. VI §2 says "the report format may not present a
   low-confidence finding with the same visual/textual weight as a
   high-confidence one," and the ROADMAP calls this out explicitly as "a
   report-template constraint, enforce it in the rendering layer, not by
   author discipline." Concretely: the mapping from confidence tier to
   visual style (`_TIER_STYLES` below) lives entirely inside this function.
   A caller supplies a `confidence` float; it cannot supply, override, or
   bypass the tier-to-style mapping itself, so a low-confidence entry can
   never be constructed to render with the same visual weight as a
   high-confidence one.

These thresholds are new Phase 4 engineering values -- Article XII does not
name them, so tuning them later is not a Constitutional amendment (mirrors
the disclosure in `agents/critic.py` for `VETO_CONFIDENCE_THRESHOLD` /
`ESCALATE_CONFIDENCE_THRESHOLD`).
"""

from __future__ import annotations

import html
import uuid
from typing import Literal

from pydantic import BaseModel, Field

HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.4

# Art. VI §2 enforcement point: tier -> inline style. Not exposed for a caller
# to configure -- see module docstring.
_TIER_STYLES: dict[str, str] = {
    "high": "font-size:1.15em; opacity:1.0; font-weight:600;",
    "medium": "font-size:1.0em; opacity:0.85; font-weight:400;",
    "low": "font-size:0.85em; opacity:0.6; font-weight:400; font-style:italic;",
}

_TIER_LABELS: dict[str, str] = {
    "high": "High confidence",
    "medium": "Medium confidence",
    "low": "Low confidence -- interpret with caution",
}


class ReportClaimInput(BaseModel):
    """Plain, decoupled input shape a caller constructs from an
    `AnalystClaim` (or any future claim-producing agent). Deliberately does
    not import `agents/analyst.py::AnalystClaim` -- see module docstring.
    """

    claim_id: str
    cell_type: str | None = None
    spatial_domain: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimReportEntry(BaseModel):
    claim_id: str
    claim_type: Literal[
        "cell_type_call",
        "spatial_domain_definition",
        "biomarker_association",
        "niche_enrichment",
        "pathway_enrichment",
    ]
    value: str | None
    confidence: float
    weight_tier: Literal["high", "medium", "low"]
    heatmap_uri: str | None = None
    shap_top_genes: list[dict] | None = None
    citations: list[dict] | None = None
    no_literature_retrieved: bool = False


class Report(BaseModel):
    run_id: uuid.UUID
    task_id: uuid.UUID
    entries: list[ClaimReportEntry]


def _tier_for(confidence: float) -> Literal["high", "medium", "low"]:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _claim_type_for(claim: ReportClaimInput) -> str:
    if claim.cell_type:
        return "cell_type_call"
    if claim.spatial_domain:
        return "spatial_domain_definition"
    # Today's claim shape (Art. XII §3 / `AnalystClaim`) only carries
    # cell_type and spatial_domain as typed fields; a claim carrying neither
    # defaults to a biomarker association rather than raising.
    return "biomarker_association"


def render_report(
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    claims: list[ReportClaimInput],
    evidence_by_claim_id: dict[str, dict] | None = None,
) -> Report:
    evidence_by_claim_id = evidence_by_claim_id or {}
    entries = []
    for claim in claims:
        evidence = evidence_by_claim_id.get(claim.claim_id, {})
        entries.append(
            ClaimReportEntry(
                claim_id=claim.claim_id,
                claim_type=_claim_type_for(claim),
                value=claim.cell_type or claim.spatial_domain,
                confidence=claim.confidence,
                weight_tier=_tier_for(claim.confidence),
                heatmap_uri=evidence.get("heatmap_uri"),
                shap_top_genes=evidence.get("shap_top_genes"),
                citations=evidence.get("citations"),
                no_literature_retrieved=evidence.get("no_literature_retrieved", False),
            )
        )
    return Report(run_id=run_id, task_id=task_id, entries=entries)


def _render_shap_item(item: dict) -> str:
    gene = item.get("gene")
    importance = item.get("importance", item.get("shap_value"))
    if gene is not None and importance is not None:
        return f"<li>{html.escape(str(gene))}: {html.escape(str(importance))}</li>"
    return f"<li>{html.escape(str(item))}</li>"


def _render_citation_item(item: dict) -> str:
    label = item.get("title") or item.get("source") or str(item)
    return f"<li>{html.escape(str(label))}</li>"


def _render_entry(entry: ClaimReportEntry) -> str:
    style = _TIER_STYLES[entry.weight_tier]
    tier_label = _TIER_LABELS[entry.weight_tier]
    value = html.escape(entry.value) if entry.value is not None else "(unspecified)"
    confidence_pct = f"{entry.confidence * 100:.1f}%"

    if entry.heatmap_uri:
        visual_html = f'<img src="{html.escape(entry.heatmap_uri)}" alt="visual attribution heatmap">'
    else:
        visual_html = "<p>no visual attribution available</p>"

    if entry.shap_top_genes:
        genes_html = "<ul>" + "".join(_render_shap_item(g) for g in entry.shap_top_genes) + "</ul>"
    else:
        genes_html = "<p>no gene-importance evidence available</p>"

    if entry.citations:
        citations_html = "<ul>" + "".join(_render_citation_item(c) for c in entry.citations) + "</ul>"
    elif entry.no_literature_retrieved:
        citations_html = "<p>no supporting literature retrieved</p>"
    else:
        citations_html = "<p>no supporting literature available</p>"

    return f"""<div class="claim-entry tier-{entry.weight_tier}" style="{style}">
  <span class="tier-badge">[{html.escape(tier_label)}]</span>
  <p class="claim-type">Claim type: {html.escape(entry.claim_type)}</p>
  <p class="claim-value">Value: {value}</p>
  <p class="claim-confidence">Confidence: {confidence_pct}</p>
  <div class="visual-attribution">{visual_html}</div>
  <div class="gene-importance">{genes_html}</div>
  <div class="citations">{citations_html}</div>
</div>"""


def render_report_html(report: Report) -> str:
    """Render a `Report` to a minimal, self-contained HTML document.

    Art. VI §2 enforcement: the tier -> style mapping (`_TIER_STYLES`) is
    applied here, unconditionally, per entry -- it is not a parameter of
    `Report`/`ClaimReportEntry` and cannot be supplied or overridden by a
    caller. A low-confidence entry is therefore structurally incapable of
    rendering with the same visual weight as a high-confidence one.
    """

    entries_html = "\n".join(_render_entry(entry) for entry in report.entries)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NEXUS-Agent Report {report.run_id}</title>
</head>
<body>
<h1>NEXUS-Agent Report</h1>
<p>Run: {report.run_id} / Task: {report.task_id}</p>
{entries_html}
</body>
</html>
"""
