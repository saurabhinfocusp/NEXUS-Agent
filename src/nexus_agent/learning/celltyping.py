"""Trainable cell-type head + checkpoint (de)serialization (Art. VII §2,
Art. XII §6).

`CellTypeClassifier` consumes the 128-dim fused representation
(`analyst/fusion.py::FUSED_DIM`) and predicts a cell-type label. Together
with the optional "refiner" layer trained by `lora_finetune.py`, this is what
`predict_cell_type` reconstructs from a persisted checkpoint at inference
time.

Checkpoint format: a plain dict
`{"label_map": dict[int, str], "classifier_state": <state_dict>,
"refiner_state": <state_dict or None>}`, serialized with `torch.save`/
`torch.load` over an in-memory `io.BytesIO()` buffer (no filesystem paths --
callers persist the resulting bytes via `data/object_store.py::put_bytes`).

The refiner, when present, is always saved and loaded as a *plain*
`nn.Linear(FUSED_DIM, FUSED_DIM)` state dict -- i.e. its effective merged
weights, not a PEFT adapter delta. See `lora_finetune.py`'s module docstring
for why: the refiner is trained directly (not through PEFT) in the
persisted-correction path, so there is no adapter to merge in the first
place; if a caller ever did wrap it with PEFT, `merge_and_unload()` before
saving would collapse it to the same plain-`nn.Linear` shape this loader
expects.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from torch import nn

from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.data.object_store import get_bytes


class CellTypeClassifier(nn.Module):
    """Single linear layer mapping a fused embedding to cell-type logits."""

    def __init__(self, num_classes: int, fused_dim: int = FUSED_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(fused_dim, num_classes)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """`fused`: (batch, fused_dim) -> (batch, num_classes) logits."""
        return self.linear(fused)


def save_checkpoint(
    classifier: CellTypeClassifier,
    label_map: dict[int, str],
    refiner: nn.Module | None,
) -> bytes:
    """Serialize `classifier` + `label_map` (+ optional `refiner`) to bytes."""
    payload: dict[str, Any] = {
        "label_map": label_map,
        "classifier_state": classifier.state_dict(),
        "refiner_state": refiner.state_dict() if refiner is not None else None,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def load_checkpoint(data: bytes) -> tuple[CellTypeClassifier, nn.Module | None, dict[int, str]]:
    """Reconstruct `(classifier, refiner_or_none, label_map)` from `save_checkpoint` bytes."""
    buffer = io.BytesIO(data)
    payload = torch.load(buffer, weights_only=False)

    label_map: dict[int, str] = {int(k): v for k, v in payload["label_map"].items()}

    classifier = CellTypeClassifier(num_classes=len(label_map))
    classifier.load_state_dict(payload["classifier_state"])
    classifier.eval()

    refiner: nn.Module | None = None
    if payload.get("refiner_state") is not None:
        refiner = nn.Linear(FUSED_DIM, FUSED_DIM)
        refiner.load_state_dict(payload["refiner_state"])
        refiner.eval()

    return classifier, refiner, label_map


def predict_cell_type(
    fused_embedding: list[float] | np.ndarray,
    checkpoint_bytes: bytes | None,
) -> str | None:
    """Predict a cell-type label from a fused embedding using a trained
    checkpoint. Returns `None` when no checkpoint is available yet (today's
    default, unchanged behavior) -- there is nothing to predict with before
    the first fine-tuning cycle has produced (and been promoted for) a
    checkpoint.
    """
    if checkpoint_bytes is None:
        return None

    classifier, refiner, label_map = load_checkpoint(checkpoint_bytes)

    with torch.inference_mode():
        x = torch.tensor(np.asarray(fused_embedding, dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        if refiner is not None:
            x = refiner(x)
        logits = classifier(x)
        predicted_idx = int(torch.argmax(logits, dim=-1).item())

    return label_map.get(predicted_idx)


def latest_promoted_checkpoint_uri(dsn: str) -> str | None:
    """The most recently promoted `finetune_runs.checkpoint_uri`, or `None`
    if no fine-tuning run has ever been promoted.
    """
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            """
            SELECT checkpoint_uri FROM finetune_runs
            WHERE promoted = true
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    return row[0] if row is not None else None


def latest_promoted_checkpoint_bytes(dsn: str) -> bytes | None:
    """Convenience: fetch the latest promoted checkpoint's bytes directly
    from object storage, or `None` if nothing has been promoted yet.
    """
    uri = latest_promoted_checkpoint_uri(dsn)
    if uri is None:
        return None
    return get_bytes(uri)
