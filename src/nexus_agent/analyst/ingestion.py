"""Expression ingestion (Constitution Art. V §3, Art. XII §3).

"Spatial omics data shall be ingested and processed through standard,
auditable tooling (e.g., Scanpy, Squidpy) rather than bespoke undocumented
preprocessing." Total-count normalization + log1p via Scanpy, spatial k-NN
graph + PCA feature reduction via Squidpy/Scanpy, matching Art. XII §3's
node-feature spec (`expression_pca[50]`) for the fusion transformer.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc
import squidpy as sq


def normalize(adata: ad.AnnData) -> ad.AnnData:
    """Total-count normalize + log1p, in place, returning the same object."""
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def spatial_knn_graph(adata: ad.AnnData, *, k: int = 6) -> ad.AnnData:
    """Spatial k-NN graph via Squidpy (Art. XII §3: k = 6-15, platform-
    dependent). Requires `adata.obsm["spatial"]` (cell centroid coordinates).
    Writes `adata.obsp["spatial_connectivities"]`, which `analyst/fusion.py`
    reads as the fusion transformer's attention-masking adjacency.

    `k` is capped at `n_obs - 1` when the input has fewer cells than that
    (e.g. small synthetic test data or a small real tile) -- a k-NN query
    can't return more neighbors than there are other points to find.
    """
    effective_k = min(k, adata.n_obs - 1)
    sq.gr.spatial_neighbors_knn(adata, n_neighs=effective_k)
    return adata


def expression_pca(adata: ad.AnnData, *, n_comps: int = 50):
    """PCA-reduced expression features (Art. XII §3: `expression_pca[50]`).

    `n_comps` is reduced automatically when the input has fewer than
    `n_comps` cells/genes (e.g. small synthetic test data) -- PCA can't
    produce more components than `min(n_obs, n_vars) - 1`.
    """
    max_comps = min(adata.n_obs, adata.n_vars) - 1
    effective_comps = min(n_comps, max_comps)
    sc.pp.pca(adata, n_comps=effective_comps)
    pcs = adata.obsm["X_pca"]
    if effective_comps < n_comps:
        import numpy as np

        pad = np.zeros((pcs.shape[0], n_comps - effective_comps))
        pcs = np.concatenate([pcs, pad], axis=1)
    return pcs
