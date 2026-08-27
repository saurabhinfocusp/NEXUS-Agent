"""Multimodal fusion (Constitution Art. V, Art. XII §3).

"Morphological image features and spatial gene-expression profiles must be
fused into a unified per-cell representation ... before any cell-type or
spatial-domain assignment is finalized." Node features are
`concat(morphology_embedding, expression_pca[50])`, edges are the spatial
k-NN graph (`analyst/ingestion.py::spatial_knn_graph`), and the network is
2 transformer layers with 4 attention heads emitting a 128-dim fused
per-cell representation.

`GraphFusionTransformer` is hand-rolled PyTorch (`nn.TransformerEncoder`
with attention masked to the k-NN adjacency), not `torch_geometric` -- see
ROADMAP.md's Phase 3 scope notes for why. Its weights are randomly
initialized (no training data/checkpoint exists yet): this proves the
Art. XII §3 architecture is implemented correctly, not that fusion quality
has been validated -- that's future work once there's a training signal.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import nn

from nexus_agent.analyst.ingestion import expression_pca

FUSED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2


class FusedCellRecord(BaseModel):
    """Per-cell fused representation, with provenance pointers back to both
    source modalities (Art. V §2) so downstream explanations are traceable.
    """

    cell_id: str
    fused_embedding: list[float] = Field(min_length=FUSED_DIM, max_length=FUSED_DIM)
    source_image_region: dict[str, Any]
    source_expression_profile: dict[str, Any]


class GraphFusionTransformer(nn.Module):
    def __init__(self, morphology_dim: int = 1024, expression_pca_dim: int = 50) -> None:
        super().__init__()
        self.input_proj = nn.Linear(morphology_dim + expression_pca_dim, FUSED_DIM)
        layer = nn.TransformerEncoderLayer(d_model=FUSED_DIM, nhead=NUM_HEADS, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """`node_features`: (N, morphology_dim+expression_pca_dim).
        `adjacency`: (N, N) bool, True where an edge (or self-loop) exists.
        Returns (N, FUSED_DIM).
        """
        projected = self.input_proj(node_features).unsqueeze(0)  # (1, N, FUSED_DIM)
        mask = torch.zeros(adjacency.shape, dtype=projected.dtype)
        mask = mask.masked_fill(~adjacency, float("-inf"))
        fused = self.encoder(projected, mask=mask)
        return fused.squeeze(0)


def fuse(
    vision_cells: list,
    adata: ad.AnnData,
    *,
    expression_pca_dim: int = 50,
    model: GraphFusionTransformer | None = None,
) -> list[FusedCellRecord]:
    """`vision_cells` (VisionCellRecord-like, with `.cell_id`/`.embedding_vector`
    or the equivalent dict keys) must be 1:1 row-aligned with `adata.obs`.
    `adata` must already have `spatial_knn_graph()` applied (Art. XII §3's
    k-NN edges live in `adata.obsp["spatial_connectivities"]`).
    """
    if adata.n_obs != len(vision_cells):
        raise ValueError(f"vision_cells ({len(vision_cells)}) and adata.obs ({adata.n_obs}) must be 1:1 aligned")

    def _get(cell, key):
        return getattr(cell, key) if hasattr(cell, key) else cell[key]

    morphology = np.stack([np.asarray(_get(c, "embedding_vector"), dtype=np.float64) for c in vision_cells])
    pca = expression_pca(adata, n_comps=expression_pca_dim)
    node_features = torch.tensor(np.concatenate([morphology, pca], axis=1), dtype=torch.float32)

    adjacency_sparse = adata.obsp["spatial_connectivities"]
    adjacency = torch.tensor(np.asarray(adjacency_sparse.todense()) > 0)
    adjacency.fill_diagonal_(True)  # every cell attends to itself

    model = model or GraphFusionTransformer(morphology_dim=morphology.shape[1], expression_pca_dim=expression_pca_dim)
    model.eval()
    with torch.inference_mode():
        fused = model(node_features, adjacency)

    records = []
    for i, cell in enumerate(vision_cells):
        cell_id = _get(cell, "cell_id")
        records.append(
            FusedCellRecord(
                cell_id=cell_id,
                fused_embedding=fused[i].tolist(),
                source_image_region={"cell_id": cell_id, "centroid_xy": list(_get(cell, "centroid_xy"))},
                source_expression_profile={"obs_name": str(adata.obs_names[i])},
            )
        )
    return records
