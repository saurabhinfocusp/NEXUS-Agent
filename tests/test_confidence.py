"""Confidences are measured from the data, not constants (Constitution
Art. II Principle 4, Art. III §3). Each test pins the property that makes
a confidence meaningful: it is higher where the evidence is better.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from nexus_agent.agents.spatial import _neighbor_agreement
from nexus_agent.analyst.domains import MIN_CELLS_FOR_DOMAINS, UNASSESSED_CONFIDENCE, assign_domains
from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.analyst.ingestion import spatial_knn_graph
from nexus_agent.learning.celltyping import CellTypeClassifier, predict_cell_types, save_checkpoint
from nexus_agent.vision.segmentation import cell_confidences


def _two_blobs(separation: float, n_per_blob: int = 20, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, size=(n_per_blob, dim))
    b = rng.normal(0.0, 1.0, size=(n_per_blob, dim))
    b[:, 0] += separation
    return np.vstack([a, b])


# --- Analyst: spatial domains -------------------------------------------------


def test_well_separated_domains_are_recovered_with_high_confidence():
    labels, confidences, reasoning = assign_domains(_two_blobs(separation=40.0))

    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1
    assert labels[0] != labels[-1]
    assert confidences.min() > 0.8
    assert "silhouette" in reasoning[0]


def test_domain_confidence_drops_when_domains_overlap():
    _, separated, _ = assign_domains(_two_blobs(separation=40.0))
    _, overlapping, _ = assign_domains(_two_blobs(separation=1.0))

    assert overlapping.mean() < separated.mean() - 0.2
    assert ((overlapping >= 0.0) & (overlapping <= 1.0)).all()


def test_a_cell_between_two_domains_is_less_confident_than_their_cores():
    fused = _two_blobs(separation=40.0)
    boundary_cell = np.zeros((1, fused.shape[1]))
    boundary_cell[0, 0] = 19.0  # just short of halfway between the two blob centres
    _, confidences, _ = assign_domains(np.vstack([fused, boundary_cell]))

    assert confidences[-1] < confidences[:-1].min()


@pytest.mark.parametrize("n_cells", [1, MIN_CELLS_FOR_DOMAINS - 1])
def test_too_few_cells_assigns_no_domain_and_says_so(n_cells):
    labels, confidences, reasoning = assign_domains(np.random.default_rng(0).normal(size=(n_cells, 4)))

    assert labels == [None] * n_cells
    assert (confidences == UNASSESSED_CONFIDENCE).all()
    assert "too few" in reasoning[0]


def test_identical_embeddings_assign_no_domain():
    labels, _, reasoning = assign_domains(np.ones((10, 4)))

    assert labels == [None] * 10
    assert "identical" in reasoning[0]


def test_domain_assignment_is_deterministic():
    fused = _two_blobs(separation=3.0)
    first, second = assign_domains(fused), assign_domains(fused)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])


# --- Analyst: cell-type calls --------------------------------------------------


def _checkpoint_separating_first_dimension(scale: float) -> bytes:
    classifier = CellTypeClassifier(num_classes=2)
    with torch.no_grad():
        classifier.linear.weight.zero_()
        classifier.linear.bias.zero_()
        classifier.linear.weight[0, 0] = scale
        classifier.linear.weight[1, 0] = -scale
    return save_checkpoint(classifier, {0: "T cell", 1: "B cell"}, refiner=None)


def test_cell_type_confidence_is_the_softmax_probability_of_the_chosen_label():
    clear, ambiguous = np.zeros(FUSED_DIM), np.zeros(FUSED_DIM)
    clear[0], ambiguous[0] = 5.0, -0.01

    (clear_label, clear_p), (ambiguous_label, ambiguous_p) = predict_cell_types(
        [clear, ambiguous], _checkpoint_separating_first_dimension(scale=1.0)
    )

    assert (clear_label, ambiguous_label) == ("T cell", "B cell")
    assert clear_p == pytest.approx(1 / (1 + np.exp(-10.0)), abs=1e-5)
    assert ambiguous_p == pytest.approx(1 / (1 + np.exp(-0.02)), abs=1e-5)
    assert ambiguous_p < 0.51 < clear_p


def test_no_checkpoint_means_no_call_and_no_confidence():
    assert predict_cell_types(np.zeros((3, FUSED_DIM)), None) == [(None, None)] * 3


# --- Spatial: niche membership -------------------------------------------------


def _adata_in_two_regions(n_per_region: int = 10) -> AnnData:
    rng = np.random.default_rng(0)
    adata = AnnData(
        X=np.ones((2 * n_per_region, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(2 * n_per_region)]),
    )
    adata.obsm["spatial"] = np.vstack(
        [rng.random((n_per_region, 2)) * 10, rng.random((n_per_region, 2)) * 10 + 1000]
    )
    return spatial_knn_graph(adata)


def test_niche_confidence_is_high_when_neighbours_share_the_niche():
    adata = _adata_in_two_regions()
    by_region = np.array([0] * 10 + [1] * 10)

    confidences = _neighbor_agreement(adata, by_region)

    # every one of a cell's 6 neighbours shares its niche: (6 + 1) / (6 + 2)
    assert confidences == pytest.approx(np.full(20, 7 / 8))


def test_niche_confidence_is_low_when_labels_ignore_space():
    adata = _adata_in_two_regions()
    by_region = np.array([0] * 10 + [1] * 10)
    alternating = np.arange(20) % 2

    assert _neighbor_agreement(adata, alternating).mean() < _neighbor_agreement(adata, by_region).mean() - 0.2


def test_a_cell_labelled_unlike_all_its_neighbours_scores_lowest():
    adata = _adata_in_two_regions()
    labels = np.array([0] * 10 + [1] * 10)
    labels[3] = 1  # one cell in the left region carrying the right region's niche

    confidences = _neighbor_agreement(adata, labels)

    assert confidences[3] == pytest.approx(1 / 8)  # (0 + 1) / (6 + 2)
    assert confidences[3] == confidences.min()


# --- Vision: segmentation ------------------------------------------------------


def test_cell_confidence_is_the_mean_probability_inside_each_mask():
    label_mask = np.zeros((6, 6), dtype=int)
    label_mask[0:2, 0:2] = 1
    label_mask[3:6, 3:6] = 2
    probability = np.zeros((6, 6))
    probability[0:2, 0:2] = 0.9
    probability[3:6, 3:6] = 0.6
    probability[5, 5] = 0.0  # one weak pixel inside cell 2

    confidences = cell_confidences(label_mask, probability)

    assert confidences == {"cell-1": pytest.approx(0.9), "cell-2": pytest.approx(0.6 * 8 / 9)}
