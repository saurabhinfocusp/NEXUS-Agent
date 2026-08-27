"""Segmentation AJI benchmark (Constitution Art. VIII, Art. XII §7).

Computes the Aggregated Jaccard Index (Kumar et al. 2017) between CellPose
predictions and ground-truth instance masks, using StarDist's own hosted
DSB2018 demo subset (fluorescence nuclei microscopy, ~27MB, no login
required). This gives one real, measured AJI number and proves the
benchmark harness end to end -- it does NOT by itself satisfy Art. VIII
§1's multi-platform validation requirement (DSB2018 isn't H&E, mIF, IMC,
or Visium/MERFISH), and `run_aji_benchmark` reports what it measures
rather than asserting a pass/fail threshold against the Art. XII §7
production-ready bar -- that's a separate, later gate.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from nexus_agent.vision.segmentation import segment_cells

DSB2018_URL = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"


def aggregated_jaccard_index(pred: np.ndarray, gt: np.ndarray) -> float:
    """AJI between two integer instance-label masks (0 = background).

    Each ground-truth object is matched to its best-IoU predicted object
    (if any); matched intersections/unions accumulate into the numerator/
    denominator, unmatched ground-truth objects contribute their full area
    to the denominator only, and predicted objects never matched to any
    ground-truth object (over-segmentation) also add their full area to
    the denominator only.
    """
    pred_labels = [label for label in np.unique(pred) if label != 0]
    gt_labels = [label for label in np.unique(gt) if label != 0]

    if not gt_labels:
        return 1.0 if not pred_labels else 0.0

    pred_masks = {label: (pred == label) for label in pred_labels}
    used_pred: set[int] = set()
    intersection_sum = 0
    union_sum = 0

    for g in gt_labels:
        gt_mask = gt == g
        best_iou, best_label, best_inter, best_union = 0.0, None, 0, 0
        for label, pred_mask in pred_masks.items():
            inter = int(np.logical_and(gt_mask, pred_mask).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou = inter / union
            if iou > best_iou:
                best_iou, best_label, best_inter, best_union = iou, label, inter, union

        if best_label is not None:
            intersection_sum += best_inter
            union_sum += best_union
            used_pred.add(best_label)
        else:
            union_sum += int(gt_mask.sum())

    for label, pred_mask in pred_masks.items():
        if label not in used_pred:
            union_sum += int(pred_mask.sum())

    return float(intersection_sum / union_sum) if union_sum else 0.0


def fetch_dsb2018_demo(target_dir: Path | str) -> Path:
    """Download + extract the DSB2018 demo subset into `target_dir`, caching
    it (skips the download if already present). Returns the `dsb2018/`
    directory containing `test/images/` and `test/masks/`.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = target_dir / "dsb2018"
    if (dataset_dir / "test" / "images").exists():
        return dataset_dir

    zip_path = target_dir / "dsb2018.zip"
    urllib.request.urlretrieve(DSB2018_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    zip_path.unlink()
    return dataset_dir


def run_aji_benchmark(target_dir: Path | str, *, n_images: int = 3) -> dict:
    """Segment `n_images` from the DSB2018 test split with CellPose and
    measure AJI against ground truth. Returns per-image and mean scores --
    this measures and reports, it does not gate on Art. XII §7's ≥0.60
    "production" threshold.
    """
    dataset_dir = fetch_dsb2018_demo(target_dir)
    image_paths = sorted((dataset_dir / "test" / "images").glob("*.tif"))[:n_images]

    per_image = []
    for image_path in image_paths:
        mask_path = dataset_dir / "test" / "masks" / image_path.name
        image = tifffile.imread(image_path)
        gt_mask = tifffile.imread(mask_path)

        pred_mask = segment_cells(image)
        score = aggregated_jaccard_index(pred_mask, gt_mask)
        per_image.append({"image": image_path.name, "aji": score, "n_pred_cells": int(pred_mask.max())})

    mean_aji = float(np.mean([entry["aji"] for entry in per_image])) if per_image else 0.0
    return {"platform": "dsb2018-fluorescence-nuclei-demo", "n_images": len(per_image), "mean_aji": mean_aji, "per_image": per_image}
