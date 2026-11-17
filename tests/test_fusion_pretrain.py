"""Tests for analyst/pretrain.py: self-supervised fusion pretraining, the
gene-indexed expression projector, and the fusion-base id that keeps
cell-type checkpoints from being applied to embeddings they weren't
trained on. Synthetic tissue with a known structure -- these show the
mechanism works, not that any real tissue has been learned.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from nexus_agent.analyst.fusion import FUSED_DIM, GraphFusionTransformer, fuse
from nexus_agent.analyst.ingestion import normalize, spatial_knn_graph
from nexus_agent.analyst.pretrain import (
    EXPRESSION_PCA_DIM,
    LEGACY_FUSION_SPACE_ID,
    ExpressionProjector,
    load_fusion_base,
    pretrain_fusion,
    save_fusion_base,
)
from nexus_agent.learning.celltyping import CellTypeClassifier, checkpoint_fusion_space_id, save_checkpoint

MORPHOLOGY_DIM = 16
GENES = [f"GENE{i}" for i in range(30)]


def _tissue(seed: int, n_per_region: int = 15, genes: list[str] = GENES):
    """Two cell populations in two regions. Each has its own expression
    program *and* its own morphology signature, so one modality predicts
    the other -- the structure masked reconstruction can learn.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_region
    population = np.array([0] * n_per_region + [1] * n_per_region)
    counts = rng.poisson(5, size=(n, len(genes))).astype(np.float32)
    counts[population == 0, : len(genes) // 2] += 60
    counts[population == 1, len(genes) // 2 :] += 60
    adata = AnnData(X=counts, obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n)]), var=pd.DataFrame(index=genes))
    adata.obsm["spatial"] = rng.random((n, 2)) * 10 + population[:, None] * 1000
    normalize(adata)
    spatial_knn_graph(adata)
    morphology = rng.normal(0.0, 0.1, size=(n, MORPHOLOGY_DIM))
    morphology[population == 0, 0] += 2.0
    morphology[population == 1, 1] += 2.0
    return morphology, adata, population


@pytest.fixture(scope="module")
def trained():
    samples = [_tissue(seed)[:2] for seed in (0, 1)]
    return pretrain_fusion(samples, epochs=60, lr=3e-3, seed=0)


def test_pretraining_reduces_masked_reconstruction_loss(trained):
    _base, losses = trained

    assert len(losses) == 60
    # Most features in this fixture are independent noise, which no model
    # can reconstruct; only the population signal is learnable, so the loss
    # falls toward that noise floor rather than toward zero.
    assert np.mean(losses[-5:]) < 0.9 * np.mean(losses[:5])


def test_constant_features_do_not_blow_up_the_loss():
    # 30 genes -> 20 zero-padded PCA dimensions; half the morphology vector
    # zero-padded, as VGG16's 512-dim embedding is. Neither carries
    # information, so neither may dominate the (variance-scaled) loss.
    morphology, adata, _population = _tissue(0)
    padded = np.concatenate([morphology, np.zeros_like(morphology)], axis=1)

    _base, losses = pretrain_fusion([(padded, adata)], epochs=30, lr=3e-3, seed=0)

    assert losses[0] < 10  # standardized targets start near 1 per modality
    assert losses[-1] < losses[0]


def test_pretraining_is_deterministic():
    samples = [_tissue(0)[:2]]
    first, first_losses = pretrain_fusion(samples, epochs=3, seed=7)
    second, second_losses = pretrain_fusion(samples, epochs=3, seed=7)

    assert first_losses == second_losses
    assert first.base_id == second.base_id


def test_different_trainings_get_different_base_ids():
    samples = [_tissue(0)[:2]]

    assert pretrain_fusion(samples, epochs=2, seed=1)[0].base_id != pretrain_fusion(samples, epochs=2, seed=2)[0].base_id


def test_pretraining_does_not_disturb_the_global_torch_rng():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    pretrain_fusion([_tissue(0)[:2]], epochs=2, seed=0)

    assert torch.equal(torch.rand(3), expected)


def test_base_records_the_morphology_encoder_it_was_trained_on():
    base, _losses = pretrain_fusion([_tissue(0)[:2]], epochs=1, seed=0, morphology_model_version="omiclip-x")

    assert base.morphology_model_version == "omiclip-x"
    assert load_fusion_base(save_fusion_base(base)).morphology_model_version == "omiclip-x"


def test_fusion_base_round_trips_through_bytes(trained):
    base, _losses = trained
    restored = load_fusion_base(save_fusion_base(base))

    assert restored.base_id == base.base_id
    assert restored.base_id.startswith("pretrained-")
    assert restored.projector.gene_names == base.projector.gene_names
    features = torch.randn(4, MORPHOLOGY_DIM + EXPRESSION_PCA_DIM)
    adjacency = torch.ones(4, 4, dtype=torch.bool)
    with torch.inference_mode():
        assert torch.allclose(restored.model(features, adjacency), base.model(features, adjacency))


def _separation(fused: np.ndarray, population: np.ndarray) -> float:
    """Distance between the two populations' centroids, in units of the
    average within-population spread.
    """
    a, b = fused[population == 0], fused[population == 1]
    spread = (a.std(axis=0).mean() + b.std(axis=0).mean()) / 2
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) / spread)


def _fuse_matrix(morphology, adata, model, expression_features=None) -> np.ndarray:
    cells = [
        {"cell_id": f"cell-{i}", "embedding_vector": row.tolist(), "centroid_xy": (0.0, 0.0)}
        for i, row in enumerate(morphology)
    ]
    records = fuse(cells, adata, model=model, expression_features=expression_features)
    return np.array([r.fused_embedding for r in records])


def test_pretrained_base_separates_populations_in_an_unseen_sample_better_than_random_weights(trained):
    base, _losses = trained
    morphology, adata, population = _tissue(seed=99)  # not in the pretraining set

    pretrained = _fuse_matrix(morphology, adata, base.model, base.projector.transform(adata))
    torch.manual_seed(0)
    untrained_model = GraphFusionTransformer(morphology_dim=MORPHOLOGY_DIM, expression_pca_dim=EXPRESSION_PCA_DIM)
    untrained = _fuse_matrix(morphology, adata.copy(), untrained_model)

    assert pretrained.shape == (30, FUSED_DIM)
    assert _separation(pretrained, population) > _separation(untrained, population)


def test_projector_gives_the_same_features_whatever_the_gene_order(trained):
    base, _losses = trained
    _morphology, adata, _population = _tissue(seed=5)
    shuffled = adata[:, list(reversed(GENES))].copy()

    np.testing.assert_allclose(base.projector.transform(shuffled), base.projector.transform(adata), atol=1e-6)


def test_projector_tolerates_missing_genes_and_reports_coverage(trained):
    base, _losses = trained
    _morphology, adata, _population = _tissue(seed=5)
    partial = adata[:, GENES[:20]].copy()

    features = base.projector.transform(partial)

    assert features.shape == (30, EXPRESSION_PCA_DIM)
    assert np.isfinite(features).all()
    assert base.projector.gene_coverage(partial.var_names) == pytest.approx(20 / 30)
    assert base.projector.gene_coverage(["UNRELATED"]) == 0.0


def test_projector_fit_uses_only_genes_shared_by_every_sample():
    _m, full, _p = _tissue(seed=0)
    _m, fewer, _p = _tissue(seed=1, genes=GENES[:12])

    assert ExpressionProjector.fit([full, fewer]).gene_names == tuple(GENES[:12])


def test_fuse_rejects_expression_features_of_the_wrong_shape():
    morphology, adata, _population = _tissue(seed=0)

    with pytest.raises(ValueError, match="expression_features has shape"):
        _fuse_matrix(morphology, adata, None, np.zeros((30, 7)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"mask_fraction": 0.0}, "mask_fraction"), ({"mask_fraction": 1.0}, "mask_fraction")],
)
def test_pretrain_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pretrain_fusion([_tissue(0)[:2]], epochs=1, **kwargs)


def test_pretrain_rejects_misaligned_morphology():
    morphology, adata, _population = _tissue(0)

    with pytest.raises(ValueError, match="1:1 aligned"):
        pretrain_fusion([(morphology[:-1], adata)], epochs=1)


# --- fusion-base id on cell-type checkpoints ----------------------------------


def test_checkpoint_records_the_fusion_base_it_was_trained_on():
    classifier = CellTypeClassifier(num_classes=2)
    label_map = {0: "T cell", 1: "B cell"}

    assert checkpoint_fusion_space_id(save_checkpoint(classifier, label_map, None)) == LEGACY_FUSION_SPACE_ID
    stamped = save_checkpoint(classifier, label_map, None, fusion_space_id="pretrained-abc|omiclip-x")
    assert checkpoint_fusion_space_id(stamped) == "pretrained-abc|omiclip-x"


def test_checkpoints_written_before_base_ids_existed_count_as_fixed_seed():
    import io

    classifier = CellTypeClassifier(num_classes=1)
    buffer = io.BytesIO()
    torch.save({"label_map": {0: "T cell"}, "classifier_state": classifier.state_dict(), "refiner_state": None}, buffer)

    assert checkpoint_fusion_space_id(buffer.getvalue()) == LEGACY_FUSION_SPACE_ID
    assert LEGACY_FUSION_SPACE_ID == "fixed-seed-20260827|vgg16-conv5_3-zeropad1024-v1"
