"""Tests for the RAG literature layer (Constitution Art. VI §1(3), Art. XII §5).

The first test is a fast, no-Postgres unit test of the default embedder's
semantic usefulness. The second is an integration test requiring
`docker compose up -d` (Postgres reachable) -- mirrors the skip-if-unreachable
fixture pattern in `tests/test_provenance.py`.
"""

import json
from pathlib import Path

import numpy as np
import psycopg
import pytest

import nexus_agent.xai.literature_rag as literature_rag
from nexus_agent.shared.config import settings
from nexus_agent.xai.literature_rag import (
    HashingLiteratureEmbedder,
    ingest_corpus,
    retrieve_citations,
)

_STARTER_CORPUS_PATH = Path(literature_rag.__file__).parent / "data" / "starter_corpus.json"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_hashing_embedder_ranks_on_topic_sentence_highest():
    fixture = {
        "cd3": "CD3 is a T-cell marker expressed on all mature T lymphocytes.",
        "cd68": "CD68 is a macrophage marker found on monocyte-lineage cells.",
        "necrosis": "The necrotic core forms at the center of a tumor that has outgrown its blood supply.",
        "unrelated": "Batch effects across spatial-omics platforms arise from differences in staining chemistry.",
    }
    embedder = HashingLiteratureEmbedder()
    query_embedding = embedder.embed_text("which marker identifies T lymphocytes")

    similarities = {
        name: _cosine_similarity(query_embedding, embedder.embed_text(sentence))
        for name, sentence in fixture.items()
    }

    best = max(similarities, key=similarities.get)
    assert best == "cd3", f"expected 'cd3' to rank highest, got similarities={similarities}"


# Not module-scoped autouse (unlike test_provenance.py): this file mixes the
# fast unit test above with an integration test below, and an autouse
# module-scoped skip would wrongly skip the fast test too when Postgres is
# unreachable. Instead it's requested explicitly by the integration test only.
@pytest.fixture(scope="module")
def _require_postgres():
    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            pass
    except psycopg.OperationalError as exc:
        pytest.skip(
            f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port} "
            f"({exc}); run `docker compose up -d`."
        )


@pytest.mark.integration
def test_retrieve_citations_thresholds_correctly(_require_postgres):
    entries = json.loads(_STARTER_CORPUS_PATH.read_text())[:5]
    embedder = HashingLiteratureEmbedder()

    with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM literature_chunks WHERE embedding_model_version = %s", (embedder.model_version,))

    n_inserted = ingest_corpus(settings.postgres_dsn, embedder, entries=entries)
    assert n_inserted == 5

    # On-topic query (the first ingested entry is about CD3 as a T-cell marker).
    on_topic = retrieve_citations(
        settings.postgres_dsn,
        entries[0]["text"],
        embedder,
    )
    assert on_topic.no_supporting_literature_retrieved is False
    assert len(on_topic.citations) >= 1
    assert all(c.similarity >= 0.75 for c in on_topic.citations)

    # Deliberately nonsense/off-topic query: no ingested chunk should clear threshold.
    off_topic = retrieve_citations(
        settings.postgres_dsn,
        "purple bicycle quantum kazoo festival weather forecast",
        embedder,
    )
    assert off_topic.no_supporting_literature_retrieved is True
    assert off_topic.citations == []

    with psycopg.connect(settings.postgres_dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM literature_chunks WHERE embedding_model_version = %s", (embedder.model_version,))
