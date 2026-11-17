"""Shared helpers for the integration tests that seed the learning loop's
tables (`provenance_log`, `correction_log`, `finetune_runs`).

Those tables live in the shared dev Postgres, and the code under test reads
them globally: `run_finetune_cycle` trains on every usable correction,
`feedback_value_report` groups every correction by tissue type, and Analyst
applies whichever checkpoint was promoted last. A test that leaves rows
behind therefore changes what later tests, and later real runs, see -- a
promoted row pointing at a fake checkpoint once failed every real-path run
that followed it. Each test removes exactly the rows it wrote.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

import psycopg
import pytest


def active_fusion_space_id() -> str:
    """The fusion space this deployment's fine-tune cycle trains in, so
    seeded `provenance_log` rows are ones it will actually use (untagged
    rows count as `LEGACY_FUSION_SPACE_ID` and are skipped unless the
    configured encoder happens to be VGG16).
    """
    from nexus_agent.agents.analyst import active_fusion_base
    from nexus_agent.vision.embedding import configured_embedder_version

    return active_fusion_base(configured_embedder_version()).space_id


def delete_learning_rows(
    dsn: str, *, run_ids: Iterable[uuid.UUID] = (), finetune_run_ids: Iterable[int] = ()
) -> None:
    run_ids, finetune_run_ids = list(run_ids), list(finetune_run_ids)
    with psycopg.connect(dsn, autocommit=True) as conn:
        if run_ids:
            conn.execute("DELETE FROM correction_log WHERE run_id = ANY(%s)", (run_ids,))
            conn.execute("DELETE FROM provenance_log WHERE run_id = ANY(%s)", (run_ids,))
        if finetune_run_ids:
            conn.execute("DELETE FROM finetune_runs WHERE id = ANY(%s)", (finetune_run_ids,))


class SeededRows:
    """What one test wrote, for `seeded_learning_rows` to delete afterwards."""

    def __init__(self) -> None:
        self.run_ids: list[uuid.UUID] = []
        self.finetune_run_ids: list[int] = []


@pytest.fixture()
def seeded_learning_rows():
    from nexus_agent.shared.config import settings

    rows = SeededRows()
    yield rows
    delete_learning_rows(settings.postgres_dsn, run_ids=rows.run_ids, finetune_run_ids=rows.finetune_run_ids)
