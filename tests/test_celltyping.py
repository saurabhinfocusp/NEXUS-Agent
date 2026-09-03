"""Fast unit tests (no Postgres/MinIO needed): checkpoint (de)serialization
and prediction round-trip (Constitution Art. VII §2).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.learning.celltyping import (
    CellTypeClassifier,
    load_checkpoint,
    predict_cell_type,
    save_checkpoint,
)


def test_predict_cell_type_returns_none_without_checkpoint():
    """Today's default, unchanged behavior: no checkpoint -> no prediction."""
    fused = np.zeros(FUSED_DIM, dtype=np.float32)
    assert predict_cell_type(fused, None) is None


def test_save_and_load_checkpoint_round_trips_without_refiner():
    label_map = {0: "T-cell", 1: "B-cell", 2: "Macrophage"}
    classifier = CellTypeClassifier(num_classes=len(label_map))

    data = save_checkpoint(classifier, label_map, refiner=None)
    assert isinstance(data, bytes)

    restored_classifier, restored_refiner, restored_label_map = load_checkpoint(data)

    assert restored_label_map == label_map
    assert restored_refiner is None
    for p1, p2 in zip(classifier.parameters(), restored_classifier.parameters()):
        assert torch.equal(p1, p2)


def test_save_and_load_checkpoint_round_trips_with_refiner():
    label_map = {0: "T-cell", 1: "B-cell"}
    classifier = CellTypeClassifier(num_classes=len(label_map))
    refiner = nn.Linear(FUSED_DIM, FUSED_DIM)

    data = save_checkpoint(classifier, label_map, refiner=refiner)
    restored_classifier, restored_refiner, restored_label_map = load_checkpoint(data)

    assert restored_label_map == label_map
    assert restored_refiner is not None
    assert torch.allclose(refiner.weight, restored_refiner.weight)
    assert torch.allclose(refiner.bias, restored_refiner.bias)


def test_predict_cell_type_picks_the_argmax_class_and_maps_through_label_map():
    label_map = {0: "T-cell", 1: "B-cell"}
    classifier = CellTypeClassifier(num_classes=2)

    # Force a deterministic prediction: a bias vector that overwhelmingly
    # favors class index 1 ("B-cell") regardless of the (zero) input.
    with torch.no_grad():
        classifier.linear.weight.zero_()
        classifier.linear.bias[:] = torch.tensor([-10.0, 10.0])

    checkpoint_bytes = save_checkpoint(classifier, label_map, refiner=None)

    fused = np.zeros(FUSED_DIM, dtype=np.float32)
    predicted = predict_cell_type(fused, checkpoint_bytes)

    assert predicted == "B-cell"


def test_predict_cell_type_applies_refiner_when_present():
    label_map = {0: "T-cell", 1: "B-cell"}
    classifier = CellTypeClassifier(num_classes=2)
    with torch.no_grad():
        classifier.linear.weight.zero_()
        classifier.linear.bias[:] = torch.tensor([-10.0, 10.0])

    # A refiner that zeroes everything out shouldn't change the argmax here
    # since the classifier's decision is driven entirely by its bias.
    refiner = nn.Linear(FUSED_DIM, FUSED_DIM)
    with torch.no_grad():
        refiner.weight.zero_()
        refiner.bias.zero_()

    checkpoint_bytes = save_checkpoint(classifier, label_map, refiner=refiner)
    fused = np.random.default_rng(0).normal(size=FUSED_DIM).astype(np.float32)
    predicted = predict_cell_type(fused, checkpoint_bytes)

    assert predicted == "B-cell"
