"""Cross-modal agreement: does a cell's image match its own expression
profile? (Constitution Art. II Principles 2 and 4, Art. V.)

OmiCLIP (`vision/omiclip.py`) puts image patches and gene sentences in one
space. For cell i, agreement is the share of the *other* cells whose
expression profile matches i's image less well than i's own does -- a
within-sample retrieval rank. 1.0: the image picks out its own profile
over every other cell's. 0.5: chance -- the image carries no information
about which profile is its own. Ties count half.

Ranking within the sample, rather than thresholding raw cosine
similarity, keeps the score comparable across samples and tissue types,
where raw CLIP similarities shift.

Reported as evidence on each claim, not folded into its confidence: OmiCLIP
was trained on Visium spots, not single cells, and how informative this
score is for a single-cell crop has not been measured. The governance
benchmark (`benchmark/governance.py::error_detection_report`) can test it
as an error detector once labelled data is available.
"""

from __future__ import annotations

import numpy as np


def retrieval_agreement(image_latents: np.ndarray, text_latents: np.ndarray) -> np.ndarray:
    """`image_latents`, `text_latents`: (n, d), row i of each from cell i,
    L2-normalized. Returns (n,) in [0, 1]; NaN when n < 2 (nothing to rank
    against).
    """
    image_latents = np.asarray(image_latents, dtype=np.float64)
    text_latents = np.asarray(text_latents, dtype=np.float64)
    if image_latents.shape != text_latents.shape or image_latents.ndim != 2:
        raise ValueError("image_latents and text_latents must both be (n, d)")
    n = len(image_latents)
    if n < 2:
        return np.full(n, np.nan)

    similarity = image_latents @ text_latents.T  # (n images, n profiles)
    own = np.diag(similarity)[:, None]
    worse = (similarity < own).sum(axis=1).astype(np.float64)
    ties = (similarity == own).sum(axis=1) - 1  # minus the cell itself
    return (worse + 0.5 * ties) / (n - 1)
