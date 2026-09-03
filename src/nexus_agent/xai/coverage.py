"""Explainability coverage (Constitution Art. XII §7, Art. XII §8).

Art. XII §7's production threshold: "Explainability coverage -- Share of
surfaced claims carrying full Art. VI §1 artifacts, measured not sampled --
100%." This module computes that share directly from the persisted
`xai_evidence` rows (Art. XII §8: "veto rate and correction-rate trend are
first-class dashboards, not log-mining exercises ... measured, not
estimated" -- the same standard applies here), rather than sampling a subset
of claims.
"""

from __future__ import annotations

import uuid

import psycopg


def explainability_coverage(dsn: str, run_id: uuid.UUID) -> float:
    """Fraction of `run_id`'s `xai_evidence` rows whose `artifacts_present`
    is a superset of that row's own `artifacts_expected`.

    Measures, does not sample: every row for the run is checked, per Art.
    XII §7's "measured not sampled." Returns `1.0` when there are zero rows
    for the run -- a vacuous truth (there is nothing un-covered), not a
    claim that coverage was actually measured; callers deciding whether an
    empty run is a meaningful "100%" should check row count separately.
    """
    with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
        rows = conn.execute(
            "SELECT artifacts_expected, artifacts_present FROM xai_evidence WHERE run_id = %s",
            (run_id,),
        ).fetchall()

    if not rows:
        return 1.0

    fully_covered = sum(
        1 for row in rows if set(row["artifacts_present"]) >= set(row["artifacts_expected"])
    )
    return fully_covered / len(rows)
