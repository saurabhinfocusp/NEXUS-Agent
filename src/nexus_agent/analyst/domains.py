"""Spatial-domain assignment on the fused representation (Constitution
Art. III §1, Art. XII §3: "cell typing and domain assignment consume only
this fused representation").

Unsupervised: KMeans over the 128-dim fused embeddings, with the number of
domains chosen by mean silhouette. Each cell's confidence is its own
silhouette coefficient rescaled from [-1, 1] to [0, 1] -- how much closer
the cell sits to its own domain than to the nearest other one. That makes
the confidence a measured property of the assignment (Art. II Principle 4:
"the system must know what it doesn't know"), not a constant: a cell on a
domain boundary scores near 0.5, a misfit below it.

What this is not: a validated domain caller. Art. XII §7's ARI >= 0.70
against expert annotation has not been measured, and the labels
(`domain_0`, ...) are cluster ids, not anatomical names. KMeans and the
candidate range are new engineering choices, not Article XII numbers.
"""

from __future__ import annotations

import numpy as np

MIN_CELLS_FOR_DOMAINS = 6
MAX_DOMAINS = 6
# Used for a cell whose domain could not be assessed at all (too few cells
# to cluster). Mid-scale on purpose: "unknown", not "confident".
UNASSESSED_CONFIDENCE = 0.5


def assign_domains(fused: np.ndarray) -> tuple[list[str | None], np.ndarray, list[str]]:
    """`fused`: (n_cells, fused_dim). Returns `(labels, confidences,
    reasoning)`; `labels[i]` is None when no domain could be assigned.
    Deterministic for a given input.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples

    n_cells = fused.shape[0]
    unassessed = ([None] * n_cells, np.full(n_cells, UNASSESSED_CONFIDENCE))
    if n_cells < MIN_CELLS_FOR_DOMAINS:
        return (
            *unassessed,
            [f"{n_cells} cell(s) is too few to cluster into spatial domains (need >= {MIN_CELLS_FOR_DOMAINS}); none assigned"],
        )
    if np.allclose(fused, fused[0]):
        return (*unassessed, ["fused embeddings are identical across cells; no spatial domains assigned"])

    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    for k in range(2, min(MAX_DOMAINS, n_cells - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(fused)
        if len(set(labels)) < 2:
            continue
        per_cell = silhouette_samples(fused, labels)
        score = float(per_cell.mean())
        if best is None or score > best[0]:
            best = (score, k, labels, per_cell)

    if best is None:
        return (*unassessed, ["KMeans found no multi-cluster structure in the fused embeddings; no spatial domains assigned"])

    score, k, labels, per_cell = best
    confidences = np.clip((per_cell + 1.0) / 2.0, 0.0, 1.0)
    reasoning = [
        f"assigned {k} spatial domain(s) by KMeans over the fused representation "
        f"(k chosen by mean silhouette = {score:.2f}); per-cell confidence is the cell's own "
        "silhouette rescaled to [0, 1]"
    ]
    return [f"domain_{int(label)}" for label in labels], confidences, reasoning
