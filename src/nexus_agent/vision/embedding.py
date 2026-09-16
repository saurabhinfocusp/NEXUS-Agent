"""Per-cell morphology embedding (Constitution Art. XII §2).

DINOv2 ViT-L/14 is the primary embedder (native 1024-dim, matching
`VisionCellRecord.embedding_vector`'s fixed contract exactly). VGG16
`conv5_3` is the lighter-weight fallback -- its native output is 512-dim,
so it's zero-padded to the same 1024-dim contract; `embedding_model_version`
is stamped distinctly (`vgg16-conv5_3-zeropad1024-v1`) so a padded
embedding is never silently indistinguishable from a real 1024-dim one
downstream.

VGG16 is the *default* on this CPU-only dev box (weights ~528MB, fast
enough to actually test); DINOv2's weight download (~1.2GB) is lazy --
only happens if `Dinov2Embedder` is actually instantiated -- so it's wired
for real without being forced on default test runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
import torch

EMBEDDING_DIM = 1024  # must match VisionCellRecord.embedding_vector (agents/vision.py), Art. XII §2


class EmbeddingBackend(ABC):
    model_version: str

    @abstractmethod
    def embed_patch(self, patch: np.ndarray) -> np.ndarray:
        """`patch`: (H, W, 3) uint8 RGB crop -> EMBEDDING_DIM-length float vector."""


class Vgg16Embedder(EmbeddingBackend):
    """Lighter-weight fallback (Art. XII §2): conv5_3 activations,
    global-average-pooled to 512-dim and zero-padded to 1024.
    """

    model_version = "vgg16-conv5_3-zeropad1024-v1"
    _RAW_DIM = 512

    def __init__(self) -> None:
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1
        full = vgg16(weights=weights)
        full.eval()
        # features[0:30] = through conv5_3's ReLU, i.e. before the final maxpool.
        self._backbone = torch.nn.Sequential(*list(full.features.children())[:30])
        self._preprocess = weights.transforms()

    @torch.inference_mode()
    def embed_patch(self, patch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)
        tensor = self._preprocess(tensor).unsqueeze(0)
        features = self._backbone(tensor)  # (1, 512, H', W')
        pooled = features.mean(dim=[2, 3]).squeeze(0).numpy().astype(np.float64)
        return np.concatenate([pooled, np.zeros(EMBEDDING_DIM - self._RAW_DIM)])


@lru_cache(maxsize=1)
def get_vgg16_embedder() -> Vgg16Embedder:
    """Return a reused VGG16 embedder instance for all uploaded-cell inference."""
    return Vgg16Embedder()


@lru_cache(maxsize=1)
def get_dinov2_embedder() -> "Dinov2Embedder":
    """Return a reused DINOv2 embedder instance when the native 1024-dim path is selected."""
    return Dinov2Embedder()


class Dinov2Embedder(EmbeddingBackend):
    """Primary embedder per Art. XII §2 -- DINOv2 ViT-L/14, native 1024-dim.
    Not the default here (CPU-only, no GPU); weight download only happens
    on first instantiation, not at import time.
    """

    model_version = "dinov2-vitl14-v1"
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self) -> None:
        self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self._model.eval()

    @torch.inference_mode()
    def embed_patch(self, patch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).float() / 255.0
        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="bilinear")
        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        embedding = self._model(tensor)  # (1, 1024)
        return embedding.squeeze(0).numpy().astype(np.float64)
