"""Pretrain the fusion transformer on paired morphology + expression data
and write a fusion base (`nexus_agent/analyst/pretrain.py`).

Each sample is two files, row-aligned (cell/spot i in one is cell/spot i in
the other):

  <name>.morphology.npy   (n_cells, 1024) Vision embeddings
  <name>.h5ad             raw counts, with obsm["spatial"] coordinates

Usage:
    python scripts/pretrain_fusion.py SAMPLE_DIR --output fusion_base.pt
    python scripts/pretrain_fusion.py SAMPLE_DIR --upload fusion/base.pt

`--upload` puts the base in the configured object store and prints the
`FUSION_CHECKPOINT_URI` to set. Switching a deployment to a new base makes
Analyst ignore cell-type checkpoints trained on the old one until a new
checkpoint is fine-tuned and promoted.

This script does not produce the morphology embeddings -- export them from
a Vision run. Nothing here checks that the result is biologically useful;
validate a base (see `nexus_agent/benchmark/`) before deploying it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from nexus_agent.analyst.ingestion import normalize, spatial_knn_graph
from nexus_agent.analyst.pretrain import pretrain_fusion, save_fusion_base


def load_samples(sample_dir: Path) -> list[tuple[np.ndarray, ad.AnnData]]:
    samples = []
    for h5ad_path in sorted(sample_dir.glob("*.h5ad")):
        morphology_path = h5ad_path.with_suffix(".morphology.npy")
        if not morphology_path.exists():
            raise FileNotFoundError(f"{h5ad_path.name} has no matching {morphology_path.name}")
        adata = ad.read_h5ad(h5ad_path)
        if "spatial" not in adata.obsm:
            raise ValueError(f"{h5ad_path.name} has no obsm['spatial']")
        normalize(adata)
        spatial_knn_graph(adata)
        samples.append((np.load(morphology_path), adata))
    if not samples:
        raise FileNotFoundError(f"no .h5ad samples found in {sample_dir}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path, help="write the base to this local file")
    parser.add_argument("--upload", metavar="KEY", help="put the base in the object store under this key")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not (args.output or args.upload):
        parser.error("give --output and/or --upload")

    samples = load_samples(args.sample_dir)
    base, losses = pretrain_fusion(
        samples, epochs=args.epochs, lr=args.lr, mask_fraction=args.mask_fraction, seed=args.seed
    )
    data = save_fusion_base(base)
    print(f"trained on {len(samples)} sample(s), {sum(a.n_obs for _m, a in samples)} cells")
    print(f"masked reconstruction loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"fusion base id: {base.base_id} ({len(base.projector.gene_names)} genes)")

    if args.output:
        args.output.write_bytes(data)
        print(f"wrote {args.output}")
    if args.upload:
        from nexus_agent.data.object_store import put_bytes

        print(f"FUSION_CHECKPOINT_URI={put_bytes(args.upload, data)}")


if __name__ == "__main__":
    main()
