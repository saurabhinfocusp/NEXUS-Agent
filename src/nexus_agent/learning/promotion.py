"""The promotion gate (Constitution Art. XII §6).

"A new checkpoint must match or exceed the prior checkpoint's Article VIII
benchmark score on every previously-validated platform, with no more than 1
percentage point of absolute regression tolerated on any single platform,
before it may be promoted to production." `prior_scores` names the set of
previously-validated platforms; a candidate checkpoint must clear the bar on
every one of them (missing coverage on a previously-validated platform is a
failure, not a pass by omission -- you cannot claim parity on a platform you
never measured).
"""

from __future__ import annotations

import json

import psycopg
from pydantic import BaseModel


class PromotionResult(BaseModel):
    passed: bool
    per_platform: dict[str, dict]


def evaluate_promotion_gate(
    new_scores: dict[str, float],
    prior_scores: dict[str, float],
    *,
    max_regression: float = 1.0,
) -> PromotionResult:
    """Compare `new_scores` against `prior_scores` (the previously-validated
    platforms) with a `max_regression`-point absolute-regression tolerance
    (Art. XII §6 default: 1.0 percentage point).
    """
    per_platform: dict[str, dict] = {}

    for platform, prior in prior_scores.items():
        new = new_scores.get(platform)
        if new is None:
            per_platform[platform] = {
                "new": None,
                "prior": prior,
                "delta": None,
                "within_tolerance": False,
            }
            continue
        delta = new - prior
        per_platform[platform] = {
            "new": new,
            "prior": prior,
            "delta": delta,
            "within_tolerance": delta >= -max_regression,
        }

    # Platforms only present in new_scores are new coverage, not a regression
    # check (there's no prior score to regress against); note them too.
    for platform, new in new_scores.items():
        if platform not in prior_scores:
            per_platform[platform] = {
                "new": new,
                "prior": None,
                "delta": None,
                "within_tolerance": True,
            }

    passed = all(per_platform[p]["within_tolerance"] for p in prior_scores)

    return PromotionResult(passed=passed, per_platform=per_platform)


def promote_checkpoint(
    dsn: str,
    finetune_run_id: int,
    new_scores: dict[str, float],
    prior_scores: dict[str, float],
    *,
    max_regression: float = 1.0,
) -> PromotionResult:
    """Evaluate the promotion gate and persist the verdict + benchmark scores
    onto the `finetune_runs` row identified by `finetune_run_id`.
    """
    result = evaluate_promotion_gate(new_scores, prior_scores, max_regression=max_regression)

    benchmark_scores = {
        "new_scores": new_scores,
        "prior_scores": prior_scores,
        "result": result.per_platform,
    }

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "UPDATE finetune_runs SET promoted = %s, benchmark_scores = %s WHERE id = %s",
            (result.passed, json.dumps(benchmark_scores), finetune_run_id),
        )

    return result
