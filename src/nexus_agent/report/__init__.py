"""Report-assembly module (Constitution Art. VI -- Explainability and
Evidentiary Standards).

Public surface: `render_report` builds a structured `Report` from claims plus
an evidence-by-claim-id lookup, and `render_report_html` renders that
`Report` to a self-contained HTML document that enforces Art. VI §2's
confidence-weighting constraint in the rendering layer itself.
"""

from __future__ import annotations

from nexus_agent.report.build import (
    ClaimReportEntry,
    Report,
    ReportClaimInput,
    render_report,
    render_report_html,
)

__all__ = [
    "ClaimReportEntry",
    "Report",
    "ReportClaimInput",
    "render_report",
    "render_report_html",
]
