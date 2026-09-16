"""Per-claim evidence bundles (Constitution Art. VI §3, Art. XII §8).

"The explainability artifacts (heatmaps, importance scores, citations,
confidence) are stored alongside the report as first-class outputs,
retrievable independently of the narrative text, so a domain expert or
auditor can verify a claim without re-running the pipeline."

Writes into the `xai_evidence` table `db/schema.sql` already creates,
mirroring `data/provenance.py`'s exact psycopg connect/execute/autocommit +
JSONB pattern. Heatmap arrays are pushed to object storage
(`data/object_store.py::put_array`) and only the resulting `s3://` URI is
persisted in Postgres, per Art. XII §8's split between object storage
(large binaries) and Postgres (structured records).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

import numpy as np
import psycopg
from pydantic import BaseModel, Field


class ClaimEvidenceBundle(BaseModel):
    """One claim's full Art. VI §1 evidence set: a visual attribution map
    (heatmap), a gene-importance score (SHAP), and a literature citation --
    or an explicit `no_literature_retrieved` flag when Art. XII §5's cosine
    similarity ≥ 0.75 citation threshold isn't met, per Art. VI §1(3).
    """

    claim_id: str
    claim_type: Literal[
        "cell_type_call",
        "spatial_domain_definition",
        "biomarker_association",
        "niche_enrichment",
        "pathway_enrichment",
    ]
    heatmap_uri: str | None = None
    shap_top_genes: list[dict] | None = None
    citations: list[dict] | None = None
    no_literature_retrieved: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    artifacts_expected: list[str]
    artifacts_present: list[str]
    component_version: str


def store_evidence_bundle(
    dsn: str,
    run_id: uuid.UUID,
    task_id: uuid.UUID,
    bundle: ClaimEvidenceBundle,
    heatmap_array: np.ndarray | None = None,
) -> ClaimEvidenceBundle:
    """Persist `bundle` into `xai_evidence`. If `heatmap_array` is given, it's
    pushed to object storage first (`xai/{run_id}/{claim_id}/heatmap.npy`)
    and `bundle.heatmap_uri` is updated to point at it before the row is
    written, so the returned bundle always reflects what was actually stored.
    """
    if heatmap_array is not None:
        from nexus_agent.data.object_store import put_array

        uri = put_array(f"xai/{run_id}/{bundle.claim_id}/heatmap.npy", heatmap_array)
        bundle = bundle.model_copy(update={"heatmap_uri": uri})

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO xai_evidence
                (run_id, task_id, claim_id, claim_type, heatmap_uri, shap_top_genes,
                 citations, no_literature_retrieved, confidence, artifacts_expected,
                 artifacts_present, component_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                task_id,
                bundle.claim_id,
                bundle.claim_type,
                bundle.heatmap_uri,
                json.dumps(bundle.shap_top_genes) if bundle.shap_top_genes is not None else None,
                json.dumps(bundle.citations) if bundle.citations is not None else None,
                bundle.no_literature_retrieved,
                bundle.confidence,
                json.dumps(bundle.artifacts_expected),
                json.dumps(bundle.artifacts_present),
                bundle.component_version,
            ),
        )
    return bundle


def fetch_evidence_bundle(dsn: str, run_id: uuid.UUID, claim_id: str) -> ClaimEvidenceBundle | None:
    """Fetch the most recent `xai_evidence` row for `(run_id, claim_id)`,
    given only `dsn, run_id, claim_id` -- no report-generation code involved
    -- so a claim's evidence can be verified independently, per Art. VI §3.
    Returns `None` if no matching row exists.
    """
    with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        row: dict[str, Any] | None = conn.execute(
            """
            SELECT * FROM xai_evidence
            WHERE run_id = %s AND claim_id = %s
            ORDER BY id DESC LIMIT 1
            """,
            (run_id, claim_id),
        ).fetchone()

    if row is None:
        return None

    return ClaimEvidenceBundle(
        claim_id=row["claim_id"],
        claim_type=row["claim_type"],
        heatmap_uri=row["heatmap_uri"],
        shap_top_genes=row["shap_top_genes"],
        citations=row["citations"],
        no_literature_retrieved=row["no_literature_retrieved"],
        confidence=row["confidence"],
        artifacts_expected=row["artifacts_expected"],
        artifacts_present=row["artifacts_present"],
        component_version=row["component_version"],
    )
