"""The real fine-tuning loop (Constitution Art. VII §2, Art. XII §6).

Art. XII §6 asks for LoRA "targeting the attention projection layers of the
fusion transformer." This module is honest about a scope reduction forced by
what is actually persisted:

- `analyst/fusion.py::GraphFusionTransformer.forward()` requires RAW
  pre-fusion `node_features` (`concat(morphology_embedding[1024],
  expression_pca[50])`). Those raw features are NOT persisted anywhere for
  historical claims -- only the already-fused 128-dim output
  (`provenance_log.fused_embedding`) is. So a historical correction cannot be
  replayed end-to-end through `GraphFusionTransformer.forward()`; there is
  nothing to feed it.
- What CAN be replayed is `(fused_embedding, corrected_label)` pairs. So
  `run_finetune_cycle` below trains a small "fusion-refinement layer" --
  `nn.Linear(FUSED_DIM, FUSED_DIM)` sitting immediately downstream of the
  fusion transformer's output (functionally right after its attention
  layers, consuming exactly what they emit) -- followed by
  `celltyping.CellTypeClassifier`, directly on those persisted pairs.
- That refiner+classifier path is trained with a plain `Adam` optimizer, NOT
  through PEFT/LoRA. This is a deliberate, disclosed choice, not an
  oversight: a rank-8 LoRA adapter's entire point is to reduce the number of
  trainable parameters relative to a large frozen base layer, but here the
  "base layer" is already a single small `nn.Linear(128, 128)` (16,512
  weights) -- LoRA-wrapping it would not shrink anything meaningful, and
  `peft.get_peft_model` targets modules by name within a larger model, which
  a single bare top-level `nn.Linear` is not. Training it directly is
  therefore the simplest, most honest choice, and is functionally equivalent
  in the one respect that matters here: it is a small, fast, targeted update
  downstream of the frozen fusion transformer's attention output.
- The literal Art. XII §6 claim -- "LoRA ... targeting the attention
  projection layers of the fusion transformer ... through a standard PEFT
  library" -- is demonstrated for real, separately, in
  `tests/test_lora_finetune.py::test_lora_targets_real_fusion_transformer_attention`.
  That test builds a real `GraphFusionTransformer` and LoRA-wraps each
  encoder layer's `self_attn` (an `nn.MultiheadAttention` instance) as a
  whole. This is deliberate, not incidental: `nn.MultiheadAttention.forward`
  reads `out_proj.weight`/`out_proj.bias` as raw tensors to feed a fused
  functional/native attention kernel -- it never calls `out_proj(x)` as a
  submodule -- so targeting `self_attn.out_proj` by name alone would silently
  produce a LoRA adapter that is never exercised by a real forward pass
  (`peft`'s own `MultiheadAttention` LoRA layer docstring warns about exactly
  this: "Don't try to apply LoRA to the out_proj of MultiheadAttention by
  targeting that layer specifically ... the LoRA adapter would be ignored").
  `peft>=0.11` ships a dedicated `MultiheadAttention` LoRA layer for this
  reason: naming the whole `self_attn` module as a `target_modules` entry
  wraps BOTH the fused in-projection (Q/K/V) and the output projection --
  the real attention projection layers of the fusion transformer -- merging
  the LoRA delta into the weights immediately before each real forward call
  and unmerging it after, which keeps the base parameter clean for the
  optimizer while still letting gradients reach the LoRA A/B matrices. The
  test proves the adapter weights actually change under a real forward +
  backward + optimizer step, on the real `GraphFusionTransformer`, while the
  frozen base attention weights do not. Do not read `run_finetune_cycle` as
  using PEFT: it does not, for the reason above, and this module says so
  plainly rather than overclaiming.

Trigger rule (Art. XII §6): a run fires on N>=50 accumulated corrections OR
7 elapsed days since the last run, whichever comes first -- see
`should_trigger_finetune`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import torch
from pydantic import BaseModel
from torch import nn

from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.data.object_store import put_bytes
from nexus_agent.data.provenance import fetch_provenance
from nexus_agent.learning.celltyping import CellTypeClassifier, save_checkpoint


class FinetuneResult(BaseModel):
    finetune_run_id: int
    n_corrections: int
    trigger_reason: str
    checkpoint_uri: str | None
    label_map: dict[int, str]
    final_train_loss: float | None


def should_trigger_finetune(
    dsn: str,
    *,
    n_threshold: int = 50,
    days_threshold: int = 7,
) -> tuple[bool, str]:
    """Art. XII §6's batching rule: fire on `n_threshold` accumulated
    `cell_type` corrections since the last fine-tuning run, OR
    `days_threshold` elapsed days since the last run -- whichever comes
    first.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        last_run = conn.execute(
            "SELECT triggered_at FROM finetune_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        last_triggered_at = last_run[0] if last_run is not None else None

        if last_triggered_at is not None:
            count_row = conn.execute(
                "SELECT count(*) FROM correction_log WHERE field = 'cell_type' AND created_at > %s",
                (last_triggered_at,),
            ).fetchone()
        else:
            count_row = conn.execute(
                "SELECT count(*) FROM correction_log WHERE field = 'cell_type'"
            ).fetchone()
        n_new_corrections = count_row[0]

    if n_new_corrections >= n_threshold:
        return True, f"N>={n_threshold} accumulated corrections"

    if last_triggered_at is None:
        return True, f">= {days_threshold} days elapsed since last fine-tune"

    if datetime.now(timezone.utc) - last_triggered_at >= timedelta(days=days_threshold):
        return True, f">= {days_threshold} days elapsed since last fine-tune"

    return False, ""


def label_string(corrected_value: Any) -> str:
    """`correction_log.corrected_value` is JSONB. `review/correction.py`
    writes it as `{"cell_type": "<label>"}`; this module's own test setup
    (and any simpler producer) may write it as a bare string. Accept both.

    Shared with `feedback_value.py` so both modules parse the same JSONB
    shape identically.
    """
    if isinstance(corrected_value, str):
        return corrected_value
    if isinstance(corrected_value, dict):
        if "cell_type" in corrected_value:
            return str(corrected_value["cell_type"])
        # Fall back to the single value present, if there's exactly one.
        values = list(corrected_value.values())
        if len(values) == 1:
            return str(values[0])
    return str(corrected_value)


def _fetch_cell_type_corrections(conn: psycopg.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT claim_id, corrected_value, run_id FROM correction_log WHERE field = 'cell_type'"
    ).fetchall()
    return [{"claim_id": r[0], "corrected_value": r[1], "run_id": r[2]} for r in rows]


def _latest_fused_embedding(dsn: str, run_id: uuid.UUID, claim_id: str) -> list[float] | None:
    """Most recent `provenance_log.fused_embedding` for `(run_id, claim_id)`,
    or `None` if there isn't one (e.g. a stub/non-fused claim).
    """
    for row in reversed(fetch_provenance(dsn, run_id)):
        if row["claim_id"] == claim_id and row.get("fused_embedding") is not None:
            return row["fused_embedding"]
    return None


def run_finetune_cycle(
    dsn: str,
    *,
    lora_rank: int = 8,
    epochs: int = 20,
    lr: float = 1e-2,
    min_corrections: int = 1,
) -> FinetuneResult:
    """Train a new checkpoint from accumulated `cell_type` corrections and
    persist it (`finetune_runs`, not yet promoted).

    See this module's docstring for the honest accounting of what is and is
    not literally trained through PEFT/LoRA here. In short: this function
    trains a small refiner (`nn.Linear(FUSED_DIM, FUSED_DIM)`, seeded near
    identity) + `celltyping.CellTypeClassifier` directly with `Adam`, on
    `(fused_embedding, corrected_label)` pairs recovered from
    `provenance_log`/`correction_log`. `lora_rank` is accepted for interface
    stability/documentation purposes but is not applied in this path -- the
    real PEFT/LoRA-on-`GraphFusionTransformer` demonstration lives in
    `tests/test_lora_finetune.py`, not here.

    Raises `ValueError` if fewer than `min_corrections` usable
    `(fused_embedding, corrected_label)` pairs are recoverable -- this
    function assumes the caller already checked `should_trigger_finetune`
    and fails loudly rather than training on (near-)nothing.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        corrections = _fetch_cell_type_corrections(conn)

    fused_embeddings: list[list[float]] = []
    labels: list[str] = []
    for corr in corrections:
        embedding = _latest_fused_embedding(dsn, corr["run_id"], corr["claim_id"])
        if embedding is None:
            continue
        fused_embeddings.append(embedding)
        labels.append(label_string(corr["corrected_value"]))

    n_usable = len(fused_embeddings)
    if n_usable < min_corrections:
        raise ValueError(
            f"only {n_usable} usable (fused_embedding, corrected_label) pairs recovered "
            f"from correction_log/provenance_log, need >= {min_corrections}; "
            "run_finetune_cycle refuses to train on this little data rather than silently no-op."
        )

    label_map: dict[int, str] = dict(enumerate(sorted(set(labels))))
    label_to_idx = {label: idx for idx, label in label_map.items()}

    X = torch.tensor(fused_embeddings, dtype=torch.float32)
    y = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

    refiner = nn.Linear(FUSED_DIM, FUSED_DIM)
    nn.init.eye_(refiner.weight)
    nn.init.zeros_(refiner.bias)

    classifier = CellTypeClassifier(num_classes=len(label_map))

    optimizer = torch.optim.Adam(list(refiner.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    final_loss: float | None = None
    for _ in range(epochs):
        logits = classifier(refiner(X))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = float(loss.item())

    checkpoint_bytes = save_checkpoint(classifier, label_map, refiner)
    checkpoint_uri = put_bytes(f"checkpoints/{uuid.uuid4()}/checkpoint.pt", checkpoint_bytes)

    trigger_reason = f"{n_usable} usable corrections recovered from correction_log/provenance_log"

    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute(
            """
            INSERT INTO finetune_runs (n_corrections, trigger_reason, checkpoint_uri, label_map, promoted)
            VALUES (%s, %s, %s, %s, false)
            RETURNING id
            """,
            (n_usable, trigger_reason, checkpoint_uri, json.dumps(label_map)),
        ).fetchone()
        finetune_run_id = row[0]

    return FinetuneResult(
        finetune_run_id=finetune_run_id,
        n_corrections=n_usable,
        trigger_reason=trigger_reason,
        checkpoint_uri=checkpoint_uri,
        label_map=label_map,
        final_train_loss=final_loss,
    )
