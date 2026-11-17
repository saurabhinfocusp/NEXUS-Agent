"""Tests for benchmark/governance.py. Synthetic inputs with known answers:
these check the instruments measure what they claim to, not that the
pipeline scores well on anything.
"""

import numpy as np
import pytest

from nexus_agent.agents.critic import VETO_CONFIDENCE_THRESHOLD
from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.benchmark.governance import correction_learning_curve, error_detection_report


def test_confidence_that_separates_right_from_wrong_scores_perfectly():
    report = error_detection_report([0.9, 0.8, 0.95, 0.1, 0.2], [True, True, True, False, False])

    assert report.n_claims == 5 and report.n_wrong == 2
    assert report.accuracy == pytest.approx(0.6)
    assert report.threshold == VETO_CONFIDENCE_THRESHOLD
    assert report.catch_rate == 1.0
    assert report.false_alarm_rate == 0.0
    assert report.auroc == 1.0


def test_constant_confidence_carries_no_information():
    # What the pipeline reported before confidences were measured: every
    # claim at 0.7. Nothing wrong is ever caught, and AUROC is chance.
    report = error_detection_report([0.7] * 6, [True, False, True, False, True, True])

    assert report.catch_rate == 0.0
    assert report.auroc == 0.5


def test_inverted_confidence_scores_below_chance():
    report = error_detection_report([0.1, 0.2, 0.9, 0.8], [True, True, False, False])

    assert report.auroc == 0.0
    assert report.catch_rate == 0.0
    assert report.false_alarm_rate == 1.0


def test_auroc_counts_ties_as_half():
    # correct: 0.8, 0.5 / wrong: 0.5, 0.2 -> pairs: (0.8>0.5) 1, (0.8>0.2) 1, (0.5=0.5) 0.5, (0.5>0.2) 1
    report = error_detection_report([0.8, 0.5, 0.5, 0.2], [True, True, False, False])

    assert report.auroc == pytest.approx(3.5 / 4)


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    # five claims made at 0.8 confidence, four of them right
    report = error_detection_report([0.8] * 5, [True, True, True, True, False])

    assert report.expected_calibration_error == pytest.approx(0.0)


def test_calibration_error_measures_overconfidence():
    report = error_detection_report([0.95] * 4, [True, False, False, False])

    assert report.expected_calibration_error == pytest.approx(0.70)


def test_rates_are_none_when_undefined():
    all_right = error_detection_report([0.9, 0.8], [True, True])
    all_wrong = error_detection_report([0.9, 0.8], [False, False])

    assert all_right.catch_rate is None and all_right.auroc is None
    assert all_wrong.false_alarm_rate is None and all_wrong.auroc is None


@pytest.mark.parametrize(
    ("confidences", "correct"),
    [([], []), ([0.5], [True, False]), ([1.5], [True])],
)
def test_error_detection_rejects_malformed_input(confidences, correct):
    with pytest.raises(ValueError):
        error_detection_report(confidences, correct)


def _labelled_embeddings(n_per_type: int = 40):
    rng = np.random.default_rng(0)
    centres = {"T cell": 0, "B cell": 1, "Macrophage": 2}
    embeddings, labels = [], []
    for name, axis in centres.items():
        block = rng.normal(0.0, 0.05, size=(n_per_type, FUSED_DIM))
        block[:, axis] += 3.0
        embeddings.append(block)
        labels.extend([name] * n_per_type)
    return np.vstack(embeddings), labels


def test_learning_curve_rises_with_more_corrections():
    embeddings, labels = _labelled_embeddings()

    curve = correction_learning_curve(embeddings, labels, budgets=[1, 30], epochs=60)

    assert [point.n_corrections for point in curve] == [1, 30]
    # One correction teaches one label: at most that type's share of held-out cells can be right.
    assert curve[0].heldout_accuracy < 0.5
    assert curve[1].heldout_accuracy > 0.95
    assert curve[1].error_detection.n_claims == 36  # 30% of 120 cells held out


def test_learning_curve_is_reproducible():
    embeddings, labels = _labelled_embeddings()

    first = correction_learning_curve(embeddings, labels, budgets=[10], epochs=20)
    second = correction_learning_curve(embeddings, labels, budgets=[10], epochs=20)

    assert first == second


def test_learning_curve_rejects_a_budget_larger_than_the_pool():
    embeddings, labels = _labelled_embeddings(n_per_type=5)

    with pytest.raises(ValueError, match="exceeds the correction pool"):
        correction_learning_curve(embeddings, labels, budgets=[100])
