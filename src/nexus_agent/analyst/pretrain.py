"""Self-supervised pretraining of the fusion transformer (Constitution
Art. V, Art. XII §3).

`analyst/fusion.py`'s `GraphFusionTransformer` ships with random weights:
its fused output is an arbitrary projection of its inputs. This module
trains it, without labels, the way the xSiGra/SiGra family Art. XII §3
names is trained -- as a graph autoencoder. Each step hides the expression
features of a random subset of cells and the morphology features of
another, and the network must reconstruct both modalities for every cell
from the fused 128-dim representation. Recovering a hidden modality is only
possible from the cell's other modality and its spatial neighbours, which
is exactly the cross-modal, neighbourhood-aware structure fusion is meant
to capture.

A trained base is a `FusionBase`: the transformer weights plus the
`ExpressionProjector` they were trained with. The projector matters as much
as the weights. `fuse()`'s default expression features are a PCA fitted per
sample, so component 3 of one sample has nothing to do with component 3 of
another, and no fixed network -- nor any cell-type head trained on its
output -- can transfer between samples. The projector is one gene-indexed
PCA fitted on the pretraining data and applied to every later sample by
gene name.

`FusionBase.base_id` identifies the coordinate space a fused embedding
lives in. Anything trained on fused embeddings (the Phase 5 cell-type head)
is only valid for the base it was trained on; `learning/celltyping.py`
records the id in each checkpoint and Analyst refuses a mismatched one.

Not done here: no pretrained base ships with the repo, and none has been
trained on real tissue -- that needs paired image + expression data this
repo doesn't include. Until `FUSION_CHECKPOINT_URI` points at one, Analyst
keeps using the fixed-seed untrained base (`FIXED_SEED_BASE_ID`).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch
from torch import nn

from nexus_agent.analyst.fusion import FUSED_DIM, GraphFusionTransformer

EXPRESSION_PCA_DIM = 50
FIXED_SEED_BASE_ID = "fixed-seed-20260827"
MAX_PRETRAIN_THREADS = 8


def _dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=np.float64)


@dataclass(frozen=True)
class ExpressionProjector:
    """A PCA over log-normalized expression, indexed by gene name so it can
    be applied to a sample whose genes differ in order or coverage from the
    data it was fitted on.
    """

    gene_names: tuple[str, ...]
    mean: np.ndarray  # (n_genes,)
    components: np.ndarray  # (EXPRESSION_PCA_DIM, n_genes); zero rows pad a short fit

    @classmethod
    def fit(cls, adatas: list[ad.AnnData]) -> ExpressionProjector:
        """Fit on the genes shared by every sample in `adatas` (each already
        `analyst/ingestion.py::normalize`d).
        """
        from sklearn.decomposition import PCA

        shared = set(adatas[0].var_names)
        for adata in adatas[1:]:
            shared &= set(adata.var_names)
        gene_names = tuple(g for g in adatas[0].var_names if g in shared)
        if not gene_names:
            raise ValueError("the pretraining samples share no genes")

        matrix = np.vstack([_dense(adata[:, list(gene_names)].X) for adata in adatas])
        n_comps = min(EXPRESSION_PCA_DIM, matrix.shape[0] - 1, matrix.shape[1])
        if n_comps < 1:
            raise ValueError("need at least 2 cells to fit the expression projector")
        pca = PCA(n_components=n_comps, random_state=0).fit(matrix)
        components = np.zeros((EXPRESSION_PCA_DIM, len(gene_names)))
        components[:n_comps] = pca.components_
        return cls(gene_names=gene_names, mean=pca.mean_, components=components)

    def gene_coverage(self, var_names) -> float:
        """Share of the projector's genes present in a sample."""
        present = set(map(str, var_names))
        return sum(g in present for g in self.gene_names) / len(self.gene_names)

    def transform_matrix(self, matrix: np.ndarray, var_names) -> np.ndarray:
        """`matrix`: (n, len(var_names)) log-normalized expression ->
        (n, EXPRESSION_PCA_DIM). A projector gene missing from `var_names`
        is imputed at its training mean, i.e. contributes nothing.
        """
        column = {str(name): i for i, name in enumerate(var_names)}
        aligned = np.tile(self.mean, (matrix.shape[0], 1))
        for j, gene in enumerate(self.gene_names):
            i = column.get(gene)
            if i is not None:
                aligned[:, j] = matrix[:, i]
        return (aligned - self.mean) @ self.components.T

    def transform(self, adata: ad.AnnData) -> np.ndarray:
        return self.transform_matrix(_dense(adata.X), adata.var_names)


@dataclass(frozen=True)
class FusionBase:
    model: GraphFusionTransformer
    projector: ExpressionProjector
    base_id: str


def _adjacency(adata: ad.AnnData) -> torch.Tensor:
    adjacency = torch.tensor(_dense(adata.obsp["spatial_connectivities"]) > 0)
    adjacency.fill_diagonal_(True)
    return adjacency


def _feature_scale(features: torch.Tensor) -> torch.Tensor:
    """Per-feature standard deviation, with constant features left unscaled
    (scale 1). Constant features are routine, not an edge case: VGG16's
    512-dim embedding is zero-padded to 1024, and a PCA fitted on fewer
    than 50 genes/cells is zero-padded too. Dividing their reconstruction
    error by a near-zero deviation would let features that carry no
    information dominate the loss by many orders of magnitude.
    """
    std = features.std(dim=0)
    return torch.where(std > 1e-6, std, torch.ones_like(std))


def _serialize(model: GraphFusionTransformer, projector: ExpressionProjector) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "model_state": model.state_dict(),
            "morphology_dim": model.input_proj.in_features - EXPRESSION_PCA_DIM,
            "gene_names": list(projector.gene_names),
            "mean": torch.tensor(projector.mean),
            "components": torch.tensor(projector.components),
        },
        buffer,
    )
    return buffer.getvalue()


def save_fusion_base(base: FusionBase) -> bytes:
    return _serialize(base.model, base.projector)


def load_fusion_base(data: bytes) -> FusionBase:
    """Rebuild a `FusionBase` from `save_fusion_base` bytes. `base_id` is
    derived from the bytes themselves, so two different trainings can never
    share an id.
    """
    payload = torch.load(io.BytesIO(data), weights_only=True)
    model = GraphFusionTransformer(
        morphology_dim=payload["morphology_dim"], expression_pca_dim=EXPRESSION_PCA_DIM
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    projector = ExpressionProjector(
        gene_names=tuple(payload["gene_names"]),
        mean=payload["mean"].numpy(),
        components=payload["components"].numpy(),
    )
    return FusionBase(model=model, projector=projector, base_id="pretrained-" + hashlib.sha256(data).hexdigest()[:16])


def pretrain_fusion(
    samples: list[tuple[np.ndarray, ad.AnnData]],
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    mask_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[FusionBase, list[float]]:
    """Train a fusion base on `samples`, each `(morphology, adata)`:
    `morphology` is (n_cells, morphology_dim), row-aligned with `adata`,
    which must already be normalized and carry its spatial k-NN graph
    (`analyst/ingestion.py`). Returns the base and the per-epoch
    reconstruction loss, measured on the hidden features only.

    Deterministic for a given `seed`. Full-batch per sample with dense
    (n_cells x n_cells) attention, so memory grows quadratically with the
    largest sample.
    """
    if not samples:
        raise ValueError("need at least one sample to pretrain on")
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be in (0, 1)")
    for morphology, adata in samples:
        if len(morphology) != adata.n_obs:
            raise ValueError("morphology rows must be 1:1 aligned with adata.obs")

    projector = ExpressionProjector.fit([adata for _m, adata in samples])
    morphology_dim = samples[0][0].shape[1]

    rng_state = torch.random.get_rng_state()
    n_threads = torch.get_num_threads()
    try:
        # Per-step tensors here are small; on a many-core host torch's
        # default thread pool spends far longer coordinating than computing.
        torch.set_num_threads(min(n_threads, MAX_PRETRAIN_THREADS))
        torch.manual_seed(seed)
        model = GraphFusionTransformer(morphology_dim=morphology_dim, expression_pca_dim=EXPRESSION_PCA_DIM)
        # Decoders exist only to give the encoder a training signal; they are
        # discarded afterwards.
        decode_expression = nn.Linear(FUSED_DIM, EXPRESSION_PCA_DIM)
        decode_morphology = nn.Linear(FUSED_DIM, morphology_dim)

        prepared = []
        for morphology, adata in samples:
            m = torch.tensor(np.asarray(morphology), dtype=torch.float32)
            e = torch.tensor(projector.transform(adata), dtype=torch.float32)
            prepared.append((m, e, _adjacency(adata)))
        # Reconstruction targets are standardized per feature, so neither
        # modality's error dominates the loss merely because its features
        # are larger or sit far from zero. The encoder still sees raw inputs.
        all_m = torch.cat([m for m, _e, _a in prepared])
        all_e = torch.cat([e for _m, e, _a in prepared])
        m_mean, m_scale = all_m.mean(dim=0), _feature_scale(all_m)
        e_mean, e_scale = all_e.mean(dim=0), _feature_scale(all_e)

        parameters = [*model.parameters(), *decode_expression.parameters(), *decode_morphology.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        generator = torch.Generator().manual_seed(seed)

        model.train()
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for m, e, adjacency in prepared:
                n = m.shape[0]
                hide_e = torch.rand(n, generator=generator) < mask_fraction
                hide_m = (torch.rand(n, generator=generator) < mask_fraction) & ~hide_e
                if not (hide_e.any() or hide_m.any()):
                    hide_e[torch.randint(n, (1,), generator=generator)] = True
                features = torch.cat(
                    [m.masked_fill(hide_m[:, None], 0.0), e.masked_fill(hide_e[:, None], 0.0)], dim=1
                )
                fused = model(features, adjacency)
                loss = torch.zeros(())
                if hide_e.any():
                    loss = loss + ((decode_expression(fused) - (e - e_mean) / e_scale)[hide_e] ** 2).mean()
                if hide_m.any():
                    loss = loss + ((decode_morphology(fused) - (m - m_mean) / m_scale)[hide_m] ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            losses.append(epoch_loss / len(prepared))

        # Rebuilt from bytes so the returned base is exactly what a later
        # `load_fusion_base` yields (same id, eval mode). Inside the guarded
        # block: constructing the model draws from the global RNG.
        model.eval()
        base = load_fusion_base(_serialize(model, projector))
    finally:
        torch.random.set_rng_state(rng_state)
        torch.set_num_threads(n_threads)

    return base, losses
