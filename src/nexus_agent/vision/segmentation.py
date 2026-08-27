"""Vision segmentation (Constitution Art. XII §2).

CellPose (`cyto3`) is the default segmenter. StarDist is Article XII's
named platform-conditioned fallback for densely packed nuclei (H&E, IMC)
but is deliberately NOT installed here -- see ROADMAP.md's Phase 3 scope
notes (it's TensorFlow-based; not worth the dependency weight for a path
this phase's dataset doesn't exercise). `stardist_segment` raises
NotImplementedError so the interface is real even though the backend isn't.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours, regionprops


def segment_cells(image: np.ndarray, *, model_type: str = "cyto3", diameter: float | None = None) -> np.ndarray:
    """Run CellPose over `image` ((H, W) or (H, W, C)), returning an integer
    label mask ((H, W), 0 = background). Import of `cellpose` is deferred so
    modules that don't segment don't pay its import cost.

    Note: as of CellPose 4.x, the library itself deprecated the separate
    `cyto`/`cyto2`/`cyto3`/`nuclei` checkpoints in favor of one unified
    "Cellpose-SAM" model (`cpsam_v2`, ~1.15GB) -- `model_type="cyto3"` is
    accepted for Art. XII §2 spec-compatibility but resolves to that same
    unified checkpoint, not a standalone cyto3-specific one. Disclosed here
    rather than silently implying a smaller/different model actually runs.
    """
    from cellpose import models

    model = models.CellposeModel(gpu=False, pretrained_model=model_type)
    masks, _flows, _styles = model.eval(image, diameter=diameter)
    return masks


def stardist_segment(image: np.ndarray, **kwargs) -> np.ndarray:
    raise NotImplementedError(
        "StarDist fallback is deferred (see ROADMAP.md Phase 3 scope notes) -- "
        "not installed on this CPU-only dev box. segment_cells() (CellPose) is the default."
    )


def cells_from_mask(label_mask: np.ndarray) -> list[dict]:
    """Per-cell centroid, boundary polygon, and bounding box from an integer
    label mask, keyed the way `agents/vision.py` needs to build
    `VisionCellRecord`s (`cell_id`, `centroid_xy`, `mask_polygon`) plus a
    `bbox` used to crop an embedding patch.
    """
    cells = []
    for region in regionprops(label_mask):
        label_only = (label_mask == region.label).astype(np.uint8)
        contours = find_contours(label_only, level=0.5)
        polygon = [(float(x), float(y)) for y, x in contours[0]] if contours else []
        y, x = region.centroid
        cells.append(
            {
                "cell_id": f"cell-{region.label}",
                "centroid_xy": (float(x), float(y)),
                "mask_polygon": polygon,
                "bbox": region.bbox,  # (min_row, min_col, max_row, max_col)
            }
        )
    return cells


def crop_patch(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop `image` to `bbox` and return an (h, w, 3) uint8 RGB patch,
    upconverting a grayscale crop by channel repetition so it's a valid
    input to an RGB-pretrained embedding backend.
    """
    min_row, min_col, max_row, max_col = bbox
    patch = image[min_row:max_row, min_col:max_col]
    if patch.ndim == 2:
        patch = np.repeat(patch[:, :, None], 3, axis=2)
    elif patch.shape[2] == 1:
        patch = np.repeat(patch, 3, axis=2)
    elif patch.shape[2] > 3:
        patch = patch[:, :, :3]
    if patch.dtype != np.uint8:
        patch = np.clip(patch, 0, 255).astype(np.uint8)
    return patch
