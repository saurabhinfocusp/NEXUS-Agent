"""Fast, Docker-free check that Art. VII §4's feedback-value report only
scores embeddings from the fusion space its checkpoint was trained in.

Postgres and object storage are monkeypatched; the end-to-end version of
this report runs in `tests/test_feedback_value.py` (integration).
"""

from __future__ import annotations

import uuid
from contextlib import nullcontext

import numpy as np

from nexus_agent.analyst.fusion import FUSED_DIM
from nexus_agent.analyst.pretrain import LEGACY_FUSION_SPACE_ID
from nexus_agent.learning import feedback_value
from nexus_agent.learning.celltyping import save_checkpoint
from nexus_agent.learning.lora_finetune import train_celltype_head

ACTIVE_SPACE = "fixed-seed-20260827|omiclip-test"


def test_report_skips_embeddings_from_another_fusion_space(monkeypatch):
    rng = np.random.default_rng(0)
    center = rng.normal(size=FUSED_DIM) * 2
    t_cell, b_cell = (center + rng.normal(size=FUSED_DIM) * 0.1), (-center + rng.normal(size=FUSED_DIM) * 0.1)
    classifier, refiner, label_map, _loss = train_celltype_head(
        [t_cell.tolist(), b_cell.tolist()], ["T-cell", "B-cell"], epochs=200, lr=0.05
    )
    checkpoint = save_checkpoint(classifier, label_map, refiner, fusion_space_id=ACTIVE_SPACE)

    in_space_run, legacy_run = uuid.uuid4(), uuid.uuid4()
    corrections = [
        {"claim_id": "a", "corrected_value": "T-cell", "run_id": in_space_run, "tissue_type": "tonsil"},
        # Same tissue, fused in the legacy (VGG16) space: its embedding shares
        # no coordinate system with the checkpoint, so it can't be scored. It
        # is deliberately the *wrong* label for where it sits, so scoring it
        # would pull accuracy_after below 1.0.
        {"claim_id": "b", "corrected_value": "B-cell", "run_id": legacy_run, "tissue_type": "tonsil"},
    ]
    provenance = {
        in_space_run: [
            {"claim_id": "a", "fused_embedding": t_cell.tolist(), "source_expression_profile": {"fusion_space_id": ACTIVE_SPACE}}
        ],
        legacy_run: [{"claim_id": "b", "fused_embedding": t_cell.tolist(), "source_expression_profile": {}}],
    }

    monkeypatch.setattr(feedback_value.psycopg, "connect", lambda *a, **k: nullcontext(None))
    monkeypatch.setattr(feedback_value, "_fetch_cell_type_corrections_with_tissue", lambda conn: corrections)
    monkeypatch.setattr(feedback_value, "latest_promoted_checkpoint_uri", lambda dsn: "s3://fake/checkpoint.pt")
    monkeypatch.setattr(feedback_value, "get_bytes", lambda uri: checkpoint)
    fake_fetch_provenance = lambda dsn, run_id: provenance[run_id]  # noqa: E731
    monkeypatch.setattr("nexus_agent.learning.lora_finetune.fetch_provenance", fake_fetch_provenance)
    monkeypatch.setattr(feedback_value, "fetch_provenance", fake_fetch_provenance, raising=False)

    report = feedback_value.feedback_value_report("postgresql://unused")

    assert LEGACY_FUSION_SPACE_ID != ACTIVE_SPACE
    assert report["tonsil"]["n_corrections"] == 2  # both counted as corrections...
    assert report["tonsil"]["accuracy_after"] == 1.0  # ...only the in-space one scored
