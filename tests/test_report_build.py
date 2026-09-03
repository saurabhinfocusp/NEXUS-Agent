"""Unit tests for the report-assembly module (Art. VI -- Explainability and
Evidentiary Standards). Fully unit-testable: no Docker/network dependency,
no `pytest.mark.integration` needed.
"""

import re
import uuid

import pytest

from nexus_agent.report.build import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    ClaimReportEntry,
    Report,
    ReportClaimInput,
    _tier_for,
    render_report,
    render_report_html,
)


# ---------------------------------------------------------------------------
# 1. Tier assignment, including both boundaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence, expected_tier",
    [
        (0.75, "high"),
        (HIGH_CONFIDENCE_THRESHOLD, "high"),  # boundary: exactly 0.7 -> high
        (0.69, "medium"),
        (0.5, "medium"),
        (MEDIUM_CONFIDENCE_THRESHOLD, "medium"),  # boundary: exactly 0.4 -> medium
        (0.39, "low"),
        (0.2, "low"),
    ],
)
def test_tier_for_boundaries(confidence, expected_tier):
    assert _tier_for(confidence) == expected_tier


@pytest.mark.parametrize(
    "confidence, expected_tier",
    [
        (0.75, "high"),
        (0.7, "high"),
        (0.69, "medium"),
        (0.5, "medium"),
        (0.4, "medium"),
        (0.39, "low"),
        (0.2, "low"),
    ],
)
def test_render_report_weight_tier_matches_tier_for(confidence, expected_tier):
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    claim = ReportClaimInput(claim_id="c0", cell_type="T-cell", confidence=confidence)
    report = render_report(run_id, task_id, [claim])
    assert report.entries[0].weight_tier == expected_tier


# ---------------------------------------------------------------------------
# 2. Rendering-layer enforcement (Art. VI §2): a low-confidence entry cannot
#    share the same visual weight as a high-confidence one, checked
#    structurally rather than by eyeballing the HTML.
# ---------------------------------------------------------------------------


def test_high_and_low_confidence_entries_render_with_different_tier_class_and_style():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    high_claim = ReportClaimInput(claim_id="high-claim", cell_type="Fibroblast", confidence=0.9)
    low_claim = ReportClaimInput(claim_id="low-claim", cell_type="Fibroblast", confidence=0.2)
    report = render_report(run_id, task_id, [high_claim, low_claim])

    assert report.entries[0].claim_type == report.entries[1].claim_type == "cell_type_call"
    assert report.entries[0].weight_tier == "high"
    assert report.entries[1].weight_tier == "low"

    html_doc = render_report_html(report)

    # Structural: locate each entry's own <div ...> opening tag (order in the
    # document matches the order of report.entries).
    div_tags = re.findall(r'<div class="claim-entry tier-[a-z]+" style="[^"]*">', html_doc)
    assert len(div_tags) == 2
    high_div, low_div = div_tags

    high_class = re.search(r'class="(claim-entry tier-[a-z]+)"', high_div).group(1)
    low_class = re.search(r'class="(claim-entry tier-[a-z]+)"', low_div).group(1)
    assert high_class != low_class
    assert high_class == "claim-entry tier-high"
    assert low_class == "claim-entry tier-low"

    high_style = re.search(r'style="([^"]*)"', high_div).group(1)
    low_style = re.search(r'style="([^"]*)"', low_div).group(1)
    assert high_style != low_style

    high_font_size = float(re.search(r"font-size:([0-9.]+)em", high_style).group(1))
    low_font_size = float(re.search(r"font-size:([0-9.]+)em", low_style).group(1))
    assert high_font_size != low_font_size
    assert high_font_size > low_font_size  # high-confidence renders with more visual weight

    high_opacity = float(re.search(r"opacity:([0-9.]+)", high_style).group(1))
    low_opacity = float(re.search(r"opacity:([0-9.]+)", low_style).group(1))
    assert high_opacity != low_opacity
    assert high_opacity > low_opacity


# ---------------------------------------------------------------------------
# 3. End-to-end render_report + evidence wiring, and HTML content check.
# ---------------------------------------------------------------------------


def test_render_report_end_to_end_with_evidence():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    claim_with_evidence = ReportClaimInput(claim_id="claim-a", cell_type="Macrophage", confidence=0.85)
    claim_without_literature = ReportClaimInput(claim_id="claim-b", spatial_domain="tumor-margin", confidence=0.3)

    evidence_by_claim_id = {
        "claim-a": {
            "heatmap_uri": "s3://bucket/heatmaps/claim-a.png",
            "shap_top_genes": [{"gene": "CD68", "importance": 0.42}],
            "citations": [{"title": "Macrophage markers in TME"}],
            "no_literature_retrieved": False,
        },
        "claim-b": {
            "heatmap_uri": None,
            "shap_top_genes": None,
            "citations": None,
            "no_literature_retrieved": True,
        },
    }

    report = render_report(run_id, task_id, [claim_with_evidence, claim_without_literature], evidence_by_claim_id)

    assert report.run_id == run_id
    assert report.task_id == task_id
    assert len(report.entries) == 2

    entry_a = report.entries[0]
    assert entry_a.claim_id == "claim-a"
    assert entry_a.claim_type == "cell_type_call"
    assert entry_a.value == "Macrophage"
    assert entry_a.confidence == 0.85
    assert entry_a.weight_tier == "high"
    assert entry_a.heatmap_uri == "s3://bucket/heatmaps/claim-a.png"
    assert entry_a.shap_top_genes == [{"gene": "CD68", "importance": 0.42}]
    assert entry_a.citations == [{"title": "Macrophage markers in TME"}]
    assert entry_a.no_literature_retrieved is False

    entry_b = report.entries[1]
    assert entry_b.claim_id == "claim-b"
    assert entry_b.claim_type == "spatial_domain_definition"
    assert entry_b.value == "tumor-margin"
    assert entry_b.confidence == 0.3
    assert entry_b.weight_tier == "low"
    assert entry_b.heatmap_uri is None
    assert entry_b.shap_top_genes is None
    assert entry_b.citations is None
    assert entry_b.no_literature_retrieved is True

    html_doc = render_report_html(report)
    assert "no supporting literature retrieved" in html_doc
    assert "s3://bucket/heatmaps/claim-a.png" in html_doc
    assert "CD68" in html_doc
    assert "Macrophage markers in TME" in html_doc


def test_render_report_defaults_evidence_when_no_lookup_given():
    run_id, task_id = uuid.uuid4(), uuid.uuid4()
    claim = ReportClaimInput(claim_id="claim-c", confidence=0.5)  # neither cell_type nor spatial_domain
    report = render_report(run_id, task_id, [claim])

    entry = report.entries[0]
    assert entry.claim_type == "biomarker_association"
    assert entry.value is None
    assert entry.weight_tier == "medium"
    assert entry.heatmap_uri is None
    assert entry.shap_top_genes is None
    assert entry.citations is None
    assert entry.no_literature_retrieved is False

    html_doc = render_report_html(report)
    assert "no visual attribution available" in html_doc
    assert "no gene-importance evidence available" in html_doc


def test_claim_report_entry_and_report_construct_directly():
    # Sanity check on the model shapes themselves, independent of render_report.
    entry = ClaimReportEntry(
        claim_id="x",
        claim_type="biomarker_association",
        value="PD-L1",
        confidence=0.6,
        weight_tier="medium",
    )
    report = Report(run_id=uuid.uuid4(), task_id=uuid.uuid4(), entries=[entry])
    assert report.entries[0].claim_id == "x"
    assert report.entries[0].no_literature_retrieved is False
