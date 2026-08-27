"""GraphFusionTransformer / fuse() (Constitution Art. XII §3).

No Docker/network needed -- weights are randomly initialized (no trained
checkpoint exists yet), so this only needs already-installed torch/scanpy/
squidpy and stays a fast test.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from nexus_agent.analyst.fusion import FUSED_DIM, GraphFusionTransformer, fuse
from nexus_agent.analyst.ingestion import spatial_knn_graph


class _FakeVisionCell:
    def __init__(self, cell_id: str, centroid_xy: tuple[float, float], embedding_vector: list[float]):
        self.cell_id = cell_id
        self.centroid_xy = centroid_xy
        self.embedding_vector = embedding_vector


def _synthetic_sample(n_cells: int = 10, n_genes: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    adata = AnnData(
        X=rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
    )
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_cells, 2))
    spatial_knn_graph(adata, k=4)

    vision_cells = [
        _FakeVisionCell(f"cell-{i}", tuple(adata.obsm["spatial"][i]), rng.normal(size=1024).tolist())
        for i in range(n_cells)
    ]
    return vision_cells, adata


def test_fuse_produces_128dim_record_per_cell_with_provenance():
    vision_cells, adata = _synthetic_sample()

    records = fuse(vision_cells, adata)

    assert len(records) == len(vision_cells)
    for record, cell in zip(records, vision_cells):
        assert len(record.fused_embedding) == FUSED_DIM
        assert record.cell_id == cell.cell_id
        assert record.source_image_region["cell_id"] == cell.cell_id
        assert record.source_expression_profile["obs_name"]


def test_fuse_rejects_misaligned_inputs():
    vision_cells, adata = _synthetic_sample(n_cells=10)
    with pytest.raises(ValueError):
        fuse(vision_cells[:5], adata)  # 5 cells vs. 10 obs rows


def test_transformer_is_deterministic_given_fixed_weights():
    vision_cells, adata = _synthetic_sample()
    torch.manual_seed(42)
    model = GraphFusionTransformer()

    records_a = fuse(vision_cells, adata, model=model)
    records_b = fuse(vision_cells, adata, model=model)

    assert records_a[0].fused_embedding == records_b[0].fused_embedding
