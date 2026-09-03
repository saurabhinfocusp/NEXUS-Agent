"""Gene-importance attribution (Constitution Art. VI §1(2), Art. XII §5).

"SHAP uses a `KernelExplainer` over the fusion layer's output (model-
agnostic, since the graph transformer is not tree-based), with a background
set of 100 randomly sampled cells per tissue sample and the top 15 genes
retained per claim."

This module only varies gene expression; the morphology half of the fused
representation is held fixed at the value for the cell being explained, so
the resulting SHAP values isolate the omics contribution per Art. VI §1(2)
("which transcriptomic features drove the decision"). PCA reduction of raw
expression is injected as `pca_transform_fn` rather than fit here, to stay
decoupled from `analyst/ingestion.py`'s PCA fit (which is per-sample, not
per-explanation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import shap
import torch
from torch import nn

if TYPE_CHECKING:
    from nexus_agent.analyst.fusion import GraphFusionTransformer


def shap_gene_importance(
    fusion_model: "GraphFusionTransformer",
    classifier_head: nn.Module,
    morphology_embedding: np.ndarray,
    background_expression: np.ndarray,
    target_expression: np.ndarray,
    gene_names: list[str],
    pca_transform_fn: Callable[[np.ndarray], np.ndarray],
    *,
    top_k: int = 15,
    n_background: int = 100,
    random_state: int = 0,
) -> list[dict]:
    """SHAP gene-importance ranking for one cell's fused-representation
    classification, holding `morphology_embedding` fixed and varying only
    gene expression.

    `fusion_model`: a real `GraphFusionTransformer` (`analyst/fusion.py`) --
    model-agnostic `KernelExplainer` is used precisely because this isn't a
    tree-based model (Art. XII §5). `classifier_head`: any injected
    `nn.Module` mapping `(batch, FUSED_DIM)` fused vectors to `(batch,
    num_classes)` logits; deliberately untyped beyond that and not imported
    from a `learning`-owned module, to avoid coupling this module to
    whichever concrete classifier Phase 5's training loop produces.

    `background_expression`: (n_cells, n_genes) from the same tissue sample;
    subsampled to `n_background` rows (Art. XII §5's "100 randomly sampled
    cells") via `np.random.default_rng(random_state)` if it has more rows
    than that, else used as-is (same small-input-capping spirit as
    `analyst/ingestion.py::spatial_knn_graph`'s k-capping).

    `target_expression`: (n_genes,) for the cell being explained.
    `pca_transform_fn`: caller-supplied `(n, n_genes) -> (n, pca_dim)`; this
    module does not fit PCA itself.

    The predicted class is fixed once from `target_expression` (under
    `torch.inference_mode()`) and every sample -- background included --
    is scored against that same class's logit, so SHAP values are
    attributions for a single, consistent target rather than each sample's
    own (possibly different) argmax.

    Returns the top `top_k` genes (fewer if `len(gene_names) < top_k`) as
    `[{"gene": name, "shap_value": float}, ...]`, sorted by descending
    `abs(shap_value)`.
    """
    rng = np.random.default_rng(random_state)
    if background_expression.shape[0] > n_background:
        idx = rng.choice(background_expression.shape[0], size=n_background, replace=False)
        background_sampled = background_expression[idx]
    else:
        background_sampled = background_expression

    morphology_tensor = np.asarray(morphology_embedding, dtype=np.float64)
    adjacency = torch.ones((1, 1), dtype=torch.bool)

    def _fused_logits(expression_row: np.ndarray) -> torch.Tensor:
        pca = pca_transform_fn(expression_row[None, :])[0]
        node_features = torch.tensor(
            np.concatenate([morphology_tensor, pca]), dtype=torch.float32
        ).unsqueeze(0)
        fused = fusion_model(node_features, adjacency)  # (1, FUSED_DIM)
        return classifier_head(fused)  # (1, num_classes)

    with torch.inference_mode():
        target_logits = _fused_logits(np.asarray(target_expression, dtype=np.float64))
        target_class = int(target_logits.argmax(dim=-1).item())

    def f(matrix: np.ndarray) -> np.ndarray:
        outputs = np.empty(matrix.shape[0], dtype=np.float64)
        with torch.inference_mode():
            for i, row in enumerate(matrix):
                logits = _fused_logits(np.asarray(row, dtype=np.float64))
                outputs[i] = logits[0, target_class].item()
        return outputs

    explainer = shap.KernelExplainer(f, background_sampled)
    shap_values = explainer.shap_values(np.asarray(target_expression, dtype=np.float64)[None, :], nsamples="auto")
    shap_values = np.asarray(shap_values).reshape(-1)  # (n_genes,)

    ranked = sorted(
        (
            {"gene": gene_names[i], "shap_value": float(shap_values[i])}
            for i in range(len(gene_names))
        ),
        key=lambda entry: abs(entry["shap_value"]),
        reverse=True,
    )
    return ranked[:top_k]
