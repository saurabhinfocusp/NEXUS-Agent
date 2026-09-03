"""RAG literature-grounding layer (Constitution Art. VI §1(3), Art. XII §5).

"The RAG layer embeds the literature corpus with a biomedical sentence
embedding model into a vector index (e.g., `pgvector`), retrieves top-k = 5
passages, and requires cosine similarity >= 0.75 for a passage to qualify as a
citation; below that threshold the claim is flagged 'no supporting literature
retrieved' per Article VI, Section 1(3), not silently uncited."

Corpus disclosure: `data/starter_corpus.json` is a small, hand-curated
starter set of ~35 general textbook-level statements about canonical
spatial-biology concepts (cell-type markers, tumor-microenvironment
vocabulary, spatial-omics methodology). It exists to prove the RAG mechanics
(embed -> index -> retrieve -> threshold -> cite-or-flag) end-to-end, and is
NOT a real literature database -- no PMIDs/DOIs are attached, and coverage is
necessarily shallow. ROADMAP.md's "Open questions" #2 (literature corpus
source: licensing and ingestion scope for a real corpus) remains open; this
module does not resolve it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel

LITERATURE_EMBEDDING_DIM = 768  # native output of pritamdeka/S-PubMedBert-MS-MARCO; matches literature_chunks.embedding (db/schema.sql)

_STARTER_CORPUS_PATH = Path(__file__).parent / "data" / "starter_corpus.json"


class LiteratureEmbeddingBackend(ABC):
    model_version: str

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """`text` -> LITERATURE_EMBEDDING_DIM-length float64 vector."""


class HashingLiteratureEmbedder(LiteratureEmbeddingBackend):
    """Lighter-weight, no-network default: a deterministic hashing-TF-IDF
    bag-of-words vectorizer (`sklearn.HashingVectorizer`), L2-normalized to
    768 dims so cosine similarity behaves sensibly.

    This is honestly a keyword-overlap stand-in, NOT a real biomedical
    semantic embedding -- it has no notion of synonymy (e.g. "T lymphocyte"
    vs. "T cell" only overlap on shared tokens) and no pretraining on
    biomedical text. It is the default here purely because it requires no
    model download and runs instantly on a CPU-only dev box, which is enough
    to exercise the retrieve -> threshold -> cite-or-flag mechanics in tests.
    `BioSentenceTransformerEmbedder` below is the named-spec real backend.
    """

    model_version = "hashing-tfidf-768-v1"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(n_features=LITERATURE_EMBEDDING_DIM, norm="l2", alternate_sign=False)

    def embed_text(self, text: str) -> np.ndarray:
        return self._vectorizer.transform([text]).toarray()[0].astype(np.float64)


class BioSentenceTransformerEmbedder(LiteratureEmbeddingBackend):
    """Real backend named by Art. XII §5: pritamdeka/S-PubMedBert-MS-MARCO,
    a PubMedBERT-based sentence embedding model fine-tuned for biomedical
    semantic search (native 768-dim output).

    Not the default here (CPU-only dev box, no GPU): weight download
    (~440MB) only happens on first instantiation, not at import time, so it
    doesn't get forced onto default test runs.
    """

    model_version = "pritamdeka-s-pubmedbert-msmarco-v1"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

    def embed_text(self, text: str) -> np.ndarray:
        embedding = self._model.encode(text, normalize_embeddings=True).astype(np.float64)
        if embedding.shape[0] != LITERATURE_EMBEDDING_DIM:
            raise ValueError(
                f"{self.model_version} produced a {embedding.shape[0]}-dim embedding, "
                f"expected {LITERATURE_EMBEDDING_DIM}; this model's assumed native "
                "dimension no longer holds -- fix the assumption, don't pad/truncate."
            )
        return embedding


def ingest_corpus(dsn: str, embedder: LiteratureEmbeddingBackend, entries: list[dict] | None = None) -> int:
    """Embed and insert literature chunks into `literature_chunks`.

    `entries` defaults to the bundled starter corpus (see module docstring).
    Note: this performs no upsert/idempotency check -- re-ingesting the same
    entries duplicates rows. Acceptable for now; callers are expected to
    ingest once per fresh database.
    """
    if entries is None:
        entries = json.loads(_STARTER_CORPUS_PATH.read_text())

    count = 0
    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        for entry in entries:
            embedding = embedder.embed_text(entry["text"])
            conn.execute(
                """
                INSERT INTO literature_chunks (text, citation, embedding, embedding_model_version)
                VALUES (%s, %s, %s, %s)
                """,
                (entry["text"], entry["citation"], embedding, embedder.model_version),
            )
            count += 1
    return count


class Citation(BaseModel):
    text: str
    citation: str
    similarity: float


class RetrievalResult(BaseModel):
    citations: list[Citation]
    no_supporting_literature_retrieved: bool


def retrieve_citations(
    dsn: str,
    query_text: str,
    embedder: LiteratureEmbeddingBackend,
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.75,
) -> RetrievalResult:
    """Retrieve top-k literature passages for `query_text` and apply the
    Art. VI §1(3) cosine-similarity >= 0.75 citation threshold.

    Below-threshold results are dropped, not returned as weak citations;
    `no_supporting_literature_retrieved` is True exactly when nothing
    survives the filter, so a claim is never silently left uncited.
    """
    query_embedding = embedder.embed_text(query_text)

    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT text, citation, 1 - (embedding <=> %s) AS similarity
            FROM literature_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        ).fetchall()

    citations = [
        Citation(text=row[0], citation=row[1], similarity=row[2])
        for row in rows
        if row[2] >= similarity_threshold
    ]
    return RetrievalResult(citations=citations, no_supporting_literature_retrieved=len(citations) == 0)
