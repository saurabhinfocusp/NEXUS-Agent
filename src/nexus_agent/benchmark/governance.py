"""Governance benchmark: does the system know when it is wrong, and what
does an expert correction buy? (Constitution Art. II Principles 3-4,
Art. VII §4, Art. VIII.)

Accuracy benchmarks (AJI, Macro-F1, ARI -- Art. XII §7) ask how often the
system is right. These ask the questions a reviewer relying on it needs
answered instead:

- `error_detection_report`: when a claim is wrong, does its confidence say
  so? Catch rate / false-alarm rate at Critic's veto threshold, AUROC of
  confidence as an error detector, and calibration error.
- `correction_learning_curve`: held-out accuracy as a function of how many
  expert corrections were fed to the fine-tune step -- the "value of expert
  feedback" Art. VII §4 requires to be measured, as a curve rather than the
  single before/after point `learning/feedback_value.py` reports for live
  data.

Everything here is a pure function of labelled inputs: it needs a dataset
with ground-truth labels, which this repo does not ship. Until it is run on
one, these are measuring instruments, not results -- no number produced
from synthetic data is evidence about real tissue.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from nexus_agent.agents.critic import VETO_CONFIDENCE_THRESHOLD


class ErrorDetectionReport(BaseModel):
    n_claims: int
    n_wrong: int
    accuracy: float
    threshold: float
    # Share of wrong claims whose confidence fell below `threshold` -- the
    # ones Critic's confidence gate would stop. None when nothing was wrong.
    catch_rate: float | None
    # Share of correct claims that fell below `threshold` anyway. None when
    # nothing was correct.
    false_alarm_rate: float | None
    # Probability that a randomly chosen correct claim has higher confidence
    # than a randomly chosen wrong one (ties count half). 0.5 = confidence
    # carries no information about correctness. None unless both exist.
    auroc: float | None
    # Expected calibration error: mean |accuracy - confidence| over
    # equal-width confidence bins, weighted by bin size. 0 = a claim made
    # with confidence p is right p of the time.
    expected_calibration_error: float


def _auroc(confidences: np.ndarray, correct: np.ndarray) -> float | None:
    positives, negatives = confidences[correct], confidences[~correct]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    # Mann-Whitney U via average ranks, so ties are counted as half.
    order = np.argsort(confidences, kind="mergesort")
    ranks = np.empty(len(confidences), dtype=np.float64)
    sorted_conf = confidences[order]
    i = 0
    while i < len(sorted_conf):
        j = i
        while j + 1 < len(sorted_conf) and sorted_conf[j + 1] == sorted_conf[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[correct].sum() - len(positives) * (len(positives) + 1) / 2.0
    return float(u / (len(positives) * len(negatives)))


def _expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    bins = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(error)


def error_detection_report(
    confidences: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
    *,
    threshold: float = VETO_CONFIDENCE_THRESHOLD,
    n_bins: int = 10,
) -> ErrorDetectionReport:
    """`confidences[i]` is the confidence the system attached to claim `i`;
    `correct[i]` is whether that claim matched ground truth. `threshold`
    defaults to Critic's veto threshold, so `catch_rate` reads as "the share
    of wrong claims the confidence gate would have stopped".
    """
    conf = np.asarray(confidences, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if conf.shape != ok.shape or conf.ndim != 1:
        raise ValueError("confidences and correct must be 1-D and the same length")
    if len(conf) == 0:
        raise ValueError("need at least one claim")
    if ((conf < 0.0) | (conf > 1.0)).any():
        raise ValueError("confidences must be in [0, 1]")

    flagged = conf < threshold
    n_wrong = int((~ok).sum())
    return ErrorDetectionReport(
        n_claims=len(conf),
        n_wrong=n_wrong,
        accuracy=float(ok.mean()),
        threshold=threshold,
        catch_rate=float(flagged[~ok].mean()) if n_wrong else None,
        false_alarm_rate=float(flagged[ok].mean()) if ok.any() else None,
        auroc=_auroc(conf, ok),
        expected_calibration_error=_expected_calibration_error(conf, ok, n_bins),
    )


class LearningCurvePoint(BaseModel):
    n_corrections: int
    heldout_accuracy: float
    # Confidence-as-error-detector quality of the head trained on this many
    # corrections, on the same held-out cells.
    error_detection: ErrorDetectionReport


def correction_learning_curve(
    fused_embeddings: np.ndarray,
    labels: list[str],
    *,
    budgets: list[int],
    heldout_fraction: float = 0.3,
    epochs: int = 20,
    lr: float = 1e-2,
    seed: int = 0,
) -> list[LearningCurvePoint]:
    """Held-out cell-typing accuracy after training on `n` expert labels,
    for each `n` in `budgets`.

    Cells are split once (seeded) into a correction pool and a held-out
    set; each budget trains a fresh head on the first `n` pool cells using
    `learning/lora_finetune.py::train_celltype_head` -- the procedure
    `run_finetune_cycle` itself uses -- and scores it on the held-out set. A
    held-out cell whose true label never appeared among the `n` corrections
    cannot be predicted and counts as wrong, which is what a reviewer would
    experience.

    This treats every correction as a labelled cell drawn at random. Real
    corrections are made where the system was wrong, so a live curve can be
    steeper or flatter than this one.
    """
    from nexus_agent.learning.celltyping import predict_cell_types, save_checkpoint
    from nexus_agent.learning.lora_finetune import train_celltype_head

    X = np.asarray(fused_embeddings, dtype=np.float64)
    if len(X) != len(labels):
        raise ValueError("fused_embeddings and labels must be the same length")
    n_heldout = int(round(len(X) * heldout_fraction))
    if n_heldout < 1 or n_heldout >= len(X):
        raise ValueError("heldout_fraction leaves no cells to train on or none to evaluate on")

    order = np.random.default_rng(seed).permutation(len(X))
    heldout, pool = order[:n_heldout], order[n_heldout:]
    if max(budgets) > len(pool):
        raise ValueError(f"largest budget ({max(budgets)}) exceeds the correction pool ({len(pool)} cells)")
    heldout_labels = [labels[i] for i in heldout]

    curve = []
    for n in budgets:
        if n < 1:
            raise ValueError("budgets must be >= 1")
        chosen = pool[:n]
        classifier, refiner, label_map, _loss = train_celltype_head(
            X[chosen].tolist(), [labels[i] for i in chosen], epochs=epochs, lr=lr, seed=seed
        )
        predictions = predict_cell_types(X[heldout], save_checkpoint(classifier, label_map, refiner))
        correct = [predicted == truth for (predicted, _p), truth in zip(predictions, heldout_labels)]
        curve.append(
            LearningCurvePoint(
                n_corrections=n,
                heldout_accuracy=float(np.mean(correct)),
                error_detection=error_detection_report([p for _label, p in predictions], correct),
            )
        )
    return curve
