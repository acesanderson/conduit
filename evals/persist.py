"""
PostgreSQL persistence layer for the eval framework.

Tables:
    documents   — corpus storage, keyed by (project, source_id),
                  with doc_hash for deduplication. Stores gold-standard
                  references alongside the source text.
    run_results — one row per (strategy, config_id, source_id) execution.
    eval_results — one row per (strategy, config_id, source_id, eval_function).
                   FK → run_results, cascades on delete.

All writes are upserts. Pool is lazily initialized and reused within a process.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Path setup: dbclients lives in a sibling project
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dbclients-project" / "src"))

from evals import RunInput, RunResult, EvalResult

_pool = None


async def _get_pool():
    global _pool
    if _pool is None:
        from dbclients.clients.postgres import get_async_pool
        _pool = await get_async_pool("evals")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_hash    TEXT        NOT NULL,
    project     TEXT        NOT NULL,
    source_id   TEXT        NOT NULL,
    text        TEXT        NOT NULL,
    reference   TEXT,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, source_id)
);
CREATE INDEX IF NOT EXISTS idx_documents_project  ON documents (project);
CREATE INDEX IF NOT EXISTS idx_documents_doc_hash ON documents (doc_hash);

CREATE TABLE IF NOT EXISTS run_results (
    strategy        TEXT        NOT NULL,
    config_id       TEXT        NOT NULL,
    source_id       TEXT        NOT NULL,
    reference_id    TEXT,
    config          JSONB       NOT NULL,
    output          TEXT        NOT NULL,
    output_metadata JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy, config_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_run_results_strategy  ON run_results (strategy);
CREATE INDEX IF NOT EXISTS idx_run_results_source_id ON run_results (source_id);

CREATE TABLE IF NOT EXISTS eval_results (
    strategy        TEXT        NOT NULL,
    config_id       TEXT        NOT NULL,
    source_id       TEXT        NOT NULL,
    eval_function   TEXT        NOT NULL,
    score           FLOAT       NOT NULL,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (strategy, config_id, source_id, eval_function),
    FOREIGN KEY (strategy, config_id, source_id)
        REFERENCES run_results (strategy, config_id, source_id)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_eval_results_strategy ON eval_results (strategy);
"""


async def ensure_tables() -> None:
    """Create all tables and indexes if they don't exist. Safe to call repeatedly."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_DDL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dump(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Documents (corpus)
# ---------------------------------------------------------------------------

async def save_corpus(project: str, items: list[RunInput]) -> None:
    """
    Upsert a list of RunInputs into the documents table under the given project tag.
    On conflict (project, source_id), updates all fields except created_at.
    """
    pool = await _get_pool()
    rows = [
        (
            _doc_hash(item.data),
            project,
            item.source_id,
            item.data,
            item.reference,
            _dump(item.metadata),
        )
        for item in items
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO documents (doc_hash, project, source_id, text, reference, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (project, source_id) DO UPDATE SET
                doc_hash  = EXCLUDED.doc_hash,
                text      = EXCLUDED.text,
                reference = EXCLUDED.reference,
                metadata  = EXCLUDED.metadata
            """,
            rows,
        )


async def load_corpus(project: str) -> list[RunInput]:
    """
    Load all documents for a project as RunInputs, ordered by source_id.
    reference is populated when a gold-standard summary exists.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_id, text, reference, metadata
            FROM   documents
            WHERE  project = $1
            ORDER  BY source_id
            """,
            project,
        )
    return [
        RunInput(
            source_id=row["source_id"],
            data=row["text"],
            reference=row["reference"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Run Results
# ---------------------------------------------------------------------------

async def save_run_results(results: list[RunResult]) -> None:
    """
    Upsert RunResults. On conflict (strategy, config_id, source_id), overwrites
    all output fields and refreshes created_at.
    """
    pool = await _get_pool()
    rows = [
        (
            r.strategy,
            r.config_id,
            r.source_id,
            r.reference_id,
            _dump(r.config if isinstance(r.config, dict) else r.config.model_dump()),
            r.output.output,
            _dump(r.output.metadata) if r.output.metadata else None,
        )
        for r in results
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO run_results
                (strategy, config_id, source_id, reference_id, config, output, output_metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb)
            ON CONFLICT (strategy, config_id, source_id) DO UPDATE SET
                reference_id    = EXCLUDED.reference_id,
                config          = EXCLUDED.config,
                output          = EXCLUDED.output,
                output_metadata = EXCLUDED.output_metadata,
                created_at      = now()
            """,
            rows,
        )


# ---------------------------------------------------------------------------
# Eval Results
# ---------------------------------------------------------------------------

async def save_eval_results(
    results: list[EvalResult],
    eval_function: str,
) -> None:
    """
    Upsert EvalResults under the given eval_function name.
    metadata may carry sub-scores by rubric domain as a nested dict.
    """
    pool = await _get_pool()
    rows = [
        (
            r.run_result.strategy,
            r.run_result.config_id,
            r.run_result.source_id,
            eval_function,
            r.score,
            None,  # metadata: add rubric dict here when EvalResult carries it
        )
        for r in results
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO eval_results
                (strategy, config_id, source_id, eval_function, score, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (strategy, config_id, source_id, eval_function) DO UPDATE SET
                score      = EXCLUDED.score,
                metadata   = EXCLUDED.metadata,
                created_at = now()
            """,
            rows,
        )
