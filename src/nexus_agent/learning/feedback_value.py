"""Benchmarking the value of the feedback loop itself (Constitution Art. VII §4).

"The improvement in accuracy attributable to expert feedback shall be
measured and reported per tissue type, so the value of human input is itself
evidence-backed, not assumed."

Framing, precisely: for each tissue type, of the cells experts corrected
(i.e. cases where the system's original output was wrong by definition --
that's *why* a correction exists), what fraction does the system now
predict correctly, using the latest promoted fine-tuned checkpoint?

- `accuracy_before` is `0.0` by construction, not a measured global accuracy
  number: a `correction_log` row only exists where a reviewer changed the
  system's output, so by definition the pre-correction prediction did not
  match the corrected value for that specific claim. This repo has no
  held-out ground-truth label set to compute a true baseline accuracy
  against, so reporting anything other than "the corrected cases were, by
  construction, wrong before correction" would be fabricating a number this
  system cannot actually measure. This is an honest simplification of a
  narrower question ("did the fine-tune fix the specific cases experts
  flagged?"), not a claim about overall model accuracy.
- `accuracy_after` replays each corrected claim's persisted
  `fused_embedding` through the latest *promoted* checkpoint and checks
  whether the prediction now matches the expert's `corrected_value`. Cases
  with no recoverable `fused_embedding` are excluded from the denominator
  (not crashed on). If no checkpoint has ever been promoted yet,
  `accuracy_after` is `None` for that tissue type -- there is nothing to
  measure post fine-tuning yet.
"""

from __future__ import annotations

from typing import Any

import psycopg

from nexus_agent.data.object_store import get_bytes
from nexus_agent.data.provenance import fetch_provenance
from nexus_agent.learning.celltyping import latest_promoted_checkpoint_uri, predict_cell_type
from nexus_agent.learning.lora_finetune import label_string


def _fetch_cell_type_corrections_with_tissue(conn: psycopg.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT claim_id, corrected_value, run_id, tissue_type FROM correction_log WHERE field = 'cell_type'"
    ).fetchall()
    return [
        {"claim_id": r[0], "corrected_value": r[1], "run_id": r[2], "tissue_type": r[3] or "unknown"}
        for r in rows
    ]


def feedback_value_report(dsn: str) -> dict[str, dict]:
    """Per-tissue-type report: `{tissue_type: {"n_corrections", "accuracy_before",
    "accuracy_after", "delta"}}`. See module docstring for the exact framing.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        corrections = _fetch_cell_type_corrections_with_tissue(conn)

    checkpoint_uri = latest_promoted_checkpoint_uri(dsn)
    checkpoint_bytes = get_bytes(checkpoint_uri) if checkpoint_uri is not None else None

    by_tissue: dict[str, list[dict[str, Any]]] = {}
    for corr in corrections:
        by_tissue.setdefault(corr["tissue_type"], []).append(corr)

    report: dict[str, dict] = {}
    for tissue_type, corrs in by_tissue.items():
        n_corrections = len(corrs)
        accuracy_before = 0.0

        if checkpoint_bytes is None:
            accuracy_after: float | None = None
        else:
            matches = 0
            n_scored = 0
            for corr in corrs:
                embedding = None
                for row in reversed(fetch_provenance(dsn, corr["run_id"])):
                    if row["claim_id"] == corr["claim_id"] and row.get("fused_embedding") is not None:
                        embedding = row["fused_embedding"]
                        break
                if embedding is None:
                    continue
                n_scored += 1
                predicted = predict_cell_type(embedding, checkpoint_bytes)
                if predicted == label_string(corr["corrected_value"]):
                    matches += 1
            accuracy_after = (matches / n_scored) if n_scored > 0 else None

        delta = (accuracy_after - accuracy_before) if accuracy_after is not None else None

        report[tissue_type] = {
            "n_corrections": n_corrections,
            "accuracy_before": accuracy_before,
            "accuracy_after": accuracy_after,
            "delta": delta,
        }

    return report
