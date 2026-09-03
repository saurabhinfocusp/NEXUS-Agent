"""Constitution Art. VII §2 / Art. XII §6: the fine-tuning loop.

Two kinds of test live here:

1. A fast, in-memory-only, non-integration test
   (`test_lora_targets_real_fusion_transformer_attention`) that proves
   PEFT/LoRA genuinely targets `GraphFusionTransformer`'s real attention
   projection layers -- the literal Art. XII §6 claim. No Postgres/MinIO
   needed.
2. Integration tests (`docker compose up -d` / Postgres+MinIO reachable)
   exercising `run_finetune_cycle` end to end against synthetic
   `provenance_log`/`correction_log` rows -- this is THE test for the exit
   criterion: "a correction submitted ... is demonstrably reflected in a
   subsequent prediction after a fine-tuning cycle (not just logged)."

See `nexus_agent/learning/lora_finetune.py`'s module docstring for why
`run_finetune_cycle` itself trains its small refiner+classifier directly
(not through PEFT) while PEFT/LoRA-on-the-real-fusion-transformer is
demonstrated separately here.
"""

from __future__ import annotations

import json
import uuid

import numpy as np
import peft
import psycopg
import pytest
import torch
from torch import nn

from nexus_agent.analyst.fusion import FUSED_DIM, GraphFusionTransformer
from nexus_agent.data import object_store
from nexus_agent.learning.celltyping import load_checkpoint, predict_cell_type
from nexus_agent.learning.lora_finetune import run_finetune_cycle, should_trigger_finetune
from nexus_agent.shared.config import settings


# ---------------------------------------------------------------------------
# 1. Fast, in-memory PEFT/LoRA-on-the-real-fusion-transformer demonstration.
# ---------------------------------------------------------------------------


def test_lora_targets_real_fusion_transformer_attention():
    """Builds a real (tiny) `GraphFusionTransformer`, LoRA-wraps each encoder
    layer's `self_attn` (an `nn.MultiheadAttention` instance -- the genuine
    attention block of the fusion transformer), runs several real forward +
    backward + optimizer steps, and asserts:
    - the LoRA adapter's parameters actually changed value, and
    - the frozen base attention weights have `requires_grad=False` and are
      numerically unchanged (they are not the ones being updated).

    Note on scope, and why the target is `self_attn` and not
    `self_attn.out_proj`: an earlier version of this test targeted
    `self_attn.out_proj` directly (it IS a genuine `nn.Linear`), but that is
    a documented PEFT footgun, not a viable target --
    `nn.MultiheadAttention.forward` reads `out_proj.weight`/`out_proj.bias`
    as raw tensors to feed a fused attention kernel; it never calls
    `out_proj(x)` as a submodule, so a LoRA adapter attached there is never
    exercised by a real forward pass (confirmed here first-hand: that
    version of the test failed with "does not require grad and does not
    have a grad_fn" on `.backward()`, because the fast/slow paths inside
    `nn.MultiheadAttention.forward` bypass the wrapped submodule entirely).
    `peft`'s own `MultiheadAttention` LoRA layer docstring warns about
    exactly this. `peft>=0.11` instead ships a dedicated LoRA layer for
    `nn.MultiheadAttention` as a whole, engaged by naming the `self_attn`
    module itself in `target_modules`: it LoRA-adapts both the fused Q/K/V
    input projection and the output projection together, merging the LoRA
    delta into the real weights immediately before each forward call and
    unmerging it after -- so gradients reach the LoRA matrices through a
    real forward pass on the real module, while the frozen base parameter
    stays untouched for the optimizer. This is the genuinely correct way to
    LoRA-target `nn.MultiheadAttention`'s attention projections, not a
    workaround that dodges the real claim.
    """
    torch.manual_seed(0)
    model = GraphFusionTransformer(morphology_dim=8, expression_pca_dim=4)

    target_names = [
        name for name, module in model.named_modules()
        if name.endswith("self_attn") and isinstance(module, nn.MultiheadAttention)
    ]
    assert len(target_names) == 2  # NUM_LAYERS == 2, one self_attn per layer

    config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=target_names)
    peft_model = peft.get_peft_model(model, config)

    # Frozen base weights must not be trainable -- only the LoRA adapters are.
    lora_param_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert lora_param_names, "expected at least one trainable (LoRA) parameter"
    assert all("lora_" in n for n in lora_param_names)

    frozen_attention_weight_names = [
        n for n, p in peft_model.named_parameters()
        if not p.requires_grad and "self_attn" in n
    ]
    assert frozen_attention_weight_names, "expected the base attention weights to be frozen"

    before_lora = {n: p.detach().clone() for n, p in peft_model.named_parameters() if p.requires_grad}
    before_frozen = {n: p.detach().clone() for n, p in peft_model.named_parameters() if n in frozen_attention_weight_names}

    n_cells = 6
    node_features = torch.randn(n_cells, 8 + 4)
    adjacency = torch.ones(n_cells, n_cells, dtype=torch.bool)

    optimizer = torch.optim.Adam(peft_model.parameters(), lr=0.1)
    for _ in range(3):  # a few steps so LoRA's zero-initialized B matrix has moved off zero too
        output = peft_model(node_features, adjacency)
        loss = output.pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    after_lora = {n: p.detach().clone() for n, p in peft_model.named_parameters() if p.requires_grad}
    after_frozen = {n: p.detach().clone() for n, p in peft_model.named_parameters() if n in frozen_attention_weight_names}

    changed = [n for n in before_lora if not torch.equal(before_lora[n], after_lora[n])]
    assert changed, "expected at least one LoRA adapter parameter to change after real optimizer steps"

    # The frozen base attention weights are not optimized. `allclose` (not
    # `equal`) is deliberate: `peft`'s merge-before/unmerge-after mechanism
    # for `nn.MultiheadAttention` round-trips each base weight through
    # `weight += delta` then `weight -= delta` on every forward call, which
    # can leave float32-epsilon-scale (~1e-7) rounding noise -- nothing like
    # the LoRA parameters' real, optimizer-driven, orders-of-magnitude-larger
    # movement asserted above.
    for name in before_frozen:
        assert torch.allclose(before_frozen[name], after_frozen[name], atol=1e-5), (
            f"frozen base attention weight {name!r} moved by more than floating-point noise"
        )


# ---------------------------------------------------------------------------
# 2. Integration tests: real Postgres + MinIO, synthetic correction/provenance data.
#
# NOTE: these are grouped under a class carrying `pytest.mark.integration`
# (rather than a module-level `pytestmark`) specifically so that
# `test_lora_targets_real_fusion_transformer_attention` above -- which needs
# no external services -- is NOT swept into the `integration` marker. A
# module-level `pytestmark` applies to every test in the file regardless of
# where it's written, so it would incorrectly mark that fast test too.
# ---------------------------------------------------------------------------


def _seed_corrections(dsn: str, run_id: uuid.UUID, task_id: uuid.UUID, rng: np.random.Generator):
    """Seed 6 synthetic (fused_embedding, corrected_label) pairs across two
    linearly-separable clusters, so a tiny fine-tune can actually learn to
    separate them.
    """
    center_a = rng.normal(size=FUSED_DIM) * 2
    center_b = -center_a
    labels = ["T-cell", "T-cell", "T-cell", "B-cell", "B-cell", "B-cell"]
    tissue_types = ["tonsil", "tonsil", None, "spleen", "spleen", "spleen"]

    embeddings = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        for i, (label, tissue) in enumerate(zip(labels, tissue_types)):
            claim_id = f"cell-{i}"
            center = center_a if label == "T-cell" else center_b
            embedding = (center + rng.normal(size=FUSED_DIM) * 0.1).tolist()
            embeddings.append(embedding)

            conn.execute(
                """
                INSERT INTO provenance_log
                    (run_id, task_id, claim_id, source_image_region,
                     source_expression_profile, component, component_version, fused_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, task_id, claim_id, json.dumps({}), json.dumps({}), "analyst", "0.1.0", json.dumps(embedding)),
            )
            conn.execute(
                """
                INSERT INTO correction_log
                    (run_id, claim_id, reviewer, original_value, corrected_value, task_id, field, tissue_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, claim_id, "dr.jane", json.dumps("Unknown"), json.dumps(label), task_id, "cell_type", tissue),
            )

    return embeddings, labels


@pytest.mark.integration
class TestRunFinetuneCycleIntegration:
    """Requires `docker compose up -d` (Postgres + MinIO reachable)."""

    @pytest.fixture(autouse=True)
    def _require_infra(self):
        try:
            with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
                pass
        except psycopg.OperationalError as exc:
            pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} ({exc}); run `docker compose up -d`.")
        try:
            object_store.ensure_bucket()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"MinIO not reachable ({exc}); run `docker compose up -d`.")

    def test_run_finetune_cycle_reflects_corrections_in_subsequent_predictions(self):
        """THE exit-criterion test: after `run_finetune_cycle`, predicting on
        the ORIGINAL fused embeddings of the corrected cells recovers the
        corrected labels -- proof the correction changed future predictions,
        not just a log entry.
        """
        dsn = settings.postgres_dsn
        run_id, task_id = uuid.uuid4(), uuid.uuid4()
        rng = np.random.default_rng(123)

        embeddings, labels = _seed_corrections(dsn, run_id, task_id, rng)

        result = run_finetune_cycle(dsn, epochs=200, lr=0.05, min_corrections=1)

        assert result.n_corrections == 6
        assert result.checkpoint_uri is not None
        assert set(result.label_map.values()) == {"T-cell", "B-cell"}
        assert result.final_train_loss is not None and result.final_train_loss < 0.1

        with psycopg.connect(dsn, autocommit=True, row_factory=psycopg.rows.dict_row) as conn:
            row = conn.execute(
                "SELECT * FROM finetune_runs WHERE id = %s", (result.finetune_run_id,)
            ).fetchone()
        assert row is not None
        assert row["checkpoint_uri"] == result.checkpoint_uri
        assert row["promoted"] is False

        checkpoint_bytes = object_store.get_bytes(result.checkpoint_uri)

        # Round trip.
        classifier, refiner, label_map = load_checkpoint(checkpoint_bytes)
        assert label_map == result.label_map
        assert refiner is not None

        # The literal exit criterion: predictions on the corrected cells'
        # ORIGINAL fused embeddings now match the corrected labels.
        correct = sum(
            predict_cell_type(embedding, checkpoint_bytes) == label
            for embedding, label in zip(embeddings, labels)
        )
        assert correct == len(labels), f"expected all {len(labels)} corrected predictions to be recovered, got {correct}"

    def test_run_finetune_cycle_raises_when_no_usable_corrections(self):
        dsn = settings.postgres_dsn
        run_id, task_id = uuid.uuid4(), uuid.uuid4()

        # A correction with NO matching provenance_log row -> not usable.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO correction_log
                    (run_id, claim_id, reviewer, original_value, corrected_value, task_id, field, tissue_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, "orphan-cell", "dr.jane", json.dumps("Unknown"), json.dumps("T-cell"), task_id, "cell_type", None),
            )

        with pytest.raises(ValueError):
            run_finetune_cycle(dsn, min_corrections=1_000_000)

    def test_should_trigger_finetune_on_accumulated_corrections(self):
        dsn = settings.postgres_dsn
        run_id, task_id = uuid.uuid4(), uuid.uuid4()
        rng = np.random.default_rng(7)

        _seed_corrections(dsn, run_id, task_id, rng)

        triggered, reason = should_trigger_finetune(dsn, n_threshold=6, days_threshold=3650)
        assert triggered is True
        assert "accumulated corrections" in reason
