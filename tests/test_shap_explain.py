"""Fast unit test: tiny dimensions so `shap.KernelExplainer` runs in a
couple of seconds on an untrained, randomly-initialized `GraphFusionTransformer`
and classifier head -- this only tests SHAP-wiring mechanics, not real gene
importance (there's no trained classifier yet, per `analyst/fusion.py`'s own
disclosure that its weights are randomly initialized).
"""

import numpy as np
import torch
from torch import nn

from nexus_agent.analyst.fusion import FUSED_DIM, GraphFusionTransformer
from nexus_agent.xai.shap_explain import shap_gene_importance


def test_shap_gene_importance_ranks_genes_by_descending_abs_value():
    rng = np.random.default_rng(0)
    n_genes = 5
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    fusion_model = GraphFusionTransformer(morphology_dim=4, expression_pca_dim=3)
    fusion_model.eval()
    classifier_head = nn.Linear(FUSED_DIM, 2)
    classifier_head.eval()

    morphology_embedding = rng.normal(size=4)
    background_expression = rng.normal(size=(8, n_genes))
    target_expression = rng.normal(size=n_genes)

    def pca_transform_fn(matrix: np.ndarray) -> np.ndarray:
        return matrix[:, :3]

    with torch.inference_mode():
        result = shap_gene_importance(
            fusion_model,
            classifier_head,
            morphology_embedding,
            background_expression,
            target_expression,
            gene_names,
            pca_transform_fn,
            top_k=15,
            n_background=8,
            random_state=0,
        )

    assert len(result) <= 15
    assert len(result) == n_genes  # fewer genes than top_k -> all returned

    for entry in result:
        assert entry["gene"] in gene_names
        assert isinstance(entry["shap_value"], float)

    abs_values = [abs(entry["shap_value"]) for entry in result]
    assert abs_values == sorted(abs_values, reverse=True)
