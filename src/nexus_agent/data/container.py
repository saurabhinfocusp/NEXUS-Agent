"""SpatialData container binding (Constitution Art. XII §1).

Binds a whole-slide/tile image, a segmentation mask, and an expression
matrix (AnnData) under one coordinate reference frame. `check_registration`
implements the cross-modal registration-error check: a sample whose max
per-cell offset meets or exceeds one cell diameter (~10-15 um) is marked
unregistered, and per Art. V §1 any claim derived from it is provisional.

Implemented generically over coordinate arrays so it is testable with
synthetic data ahead of Phase 3's real segmentation/registration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, TableModel

# One cell diameter, conservative end of the Art. XII §1 range (10-15 um).
REGISTRATION_ERROR_THRESHOLD_UM = 15.0


@dataclass(frozen=True)
class RegistrationResult:
    mean_error_um: float
    max_error_um: float
    threshold_um: float

    @property
    def registered(self) -> bool:
        return self.max_error_um < self.threshold_um


def check_registration(
    mask_centroids_um: np.ndarray,
    expression_coords_um: np.ndarray,
    threshold_um: float = REGISTRATION_ERROR_THRESHOLD_UM,
) -> RegistrationResult:
    """Per-cell offset between mask centroids and paired expression coordinates.

    Both arrays are (n_cells, 2) in the same physical units (microns) and
    1:1 paired by row (cell i's mask centroid vs. cell i's expression coord).
    """
    mask_centroids_um = np.asarray(mask_centroids_um, dtype=float)
    expression_coords_um = np.asarray(expression_coords_um, dtype=float)
    if mask_centroids_um.shape != expression_coords_um.shape:
        raise ValueError("mask_centroids_um and expression_coords_um must be 1:1 paired and same shape")
    if mask_centroids_um.ndim != 2 or mask_centroids_um.shape[1] != 2:
        raise ValueError("coordinate arrays must have shape (n_cells, 2)")

    offsets = np.linalg.norm(mask_centroids_um - expression_coords_um, axis=1)
    return RegistrationResult(
        mean_error_um=float(offsets.mean()),
        max_error_um=float(offsets.max()),
        threshold_um=threshold_um,
    )


def build_spatialdata(image: np.ndarray, mask: np.ndarray, adata: AnnData, *, labels_name: str = "cells") -> SpatialData:
    """Bind image + segmentation mask + expression matrix under one CRS.

    `image` is (c, y, x); `mask` is (y, x) integer labels. `adata.obs` must
    carry `region` (constant, equal to `labels_name`) and `instance_id`
    (matching mask label values) columns per the spatialdata TableModel
    convention, so each expression row is traceable back to its mask cell.
    """
    image_element = Image2DModel.parse(image, dims=("c", "y", "x"))
    labels_element = Labels2DModel.parse(mask, dims=("y", "x"))
    table = TableModel.parse(adata, region=labels_name, region_key="region", instance_key="instance_id")

    return SpatialData(
        images={"image": image_element},
        labels={labels_name: labels_element},
        tables={"table": table},
    )
