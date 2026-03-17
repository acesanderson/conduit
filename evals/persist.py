"""
PostgreSQL persistence layer for the eval framework.

Tables:
    configs      — deduplicated config dicts, keyed by MD5[:8] hash.
    documents    — corpus storage, keyed by (project, source_id).
    run_results  — one row per (project, strategy, config_id, source_id).
    eval_results — one row per (project, strategy, config_id, source_id, eval_function).
                   FK → run_results, cascades on delete.

Pool is lazily initialized and reused within a process.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dbclients-project" / "src"))

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
CREATE TABLE IF NOT EXISTS configs (
    config_id   TEXT        PRIMARY KEY,
    config      JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
    project         TEXT        NOT NULL,
    strategy        TEXT        NOT NULL,
    config_id       TEXT        NOT NULL REFERENCES configs (config_id),
    source_id       TEXT        NOT NULL,
    reference_id    TEXT,
    output          TEXT        NOT NULL,
    output_metadata JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, strategy, config_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_run_results_project   ON run_results (project);
CREATE INDEX IF NOT EXISTS idx_run_results_strategy  ON run_results (strategy);
CREATE INDEX IF NOT EXISTS idx_run_results_source_id ON run_results (source_id);

CREATE TABLE IF NOT EXISTS eval_results (
    project         TEXT        NOT NULL,
    strategy        TEXT        NOT NULL,
    config_id       TEXT        NOT NULL,
    source_id       TEXT        NOT NULL,
    eval_function   TEXT        NOT NULL,
    score           FLOAT       NOT NULL,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, strategy, config_id, source_id, eval_function),
    FOREIGN KEY (project, strategy, config_id, source_id)
        REFERENCES run_results (project, strategy, config_id, source_id)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_eval_results_project ON eval_results (project);
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
