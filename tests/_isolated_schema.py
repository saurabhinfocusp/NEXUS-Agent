"""Per-test copies of Postgres tables, for integration tests whose code
under test aggregates over a whole table.

`metrics/dashboards.py` computes veto rate, correction trend, latency and
confidence over all of `message_log`/`correction_log`, and
`xai/literature_rag.py` retrieves from all of `literature_chunks` -- so a
test can only hand-compute their expected output against tables holding
nothing but its own rows. Emptying the shared tables to get that would
delete real dashboard history, reviewer corrections and the RAG corpus
from the dev database, so instead each test gets a fresh schema holding
empty copies of the tables it needs, and a DSN whose `search_path`
resolves those table names there first (then `public`, for extension
types such as pgvector's `vector`). The code under test runs unchanged
and the schema is dropped afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo


@contextmanager
def isolated_tables(dsn: str, *tables: str) -> Iterator[str]:
    """Yield a DSN on which each of `tables` is an empty, schema-identical
    copy of `public.<table>`, private to the caller."""
    schema = f"nexus_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            for table in tables:
                conn.execute(
                    sql.SQL("CREATE TABLE {}.{} (LIKE {}.{} INCLUDING ALL)").format(
                        sql.Identifier(schema), sql.Identifier(table), sql.Identifier("public"), sql.Identifier(table)
                    )
                )
        except Exception:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            raise
    try:
        yield make_conninfo(dsn, options=f"-c search_path={schema},public")
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
