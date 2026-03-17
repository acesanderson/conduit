# ConduitDataset Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ConduitDataset`, a project-scoped postgres-backed handle for documents, runs, and evals, with async/sync interfaces and a `conduit-dataset` CLI.

**Architecture:** Three files: `persist.py` (raw SQL + pool singleton, updated schema), `dataset.py` (namespace classes, sync/async wrappers, exceptions), `datasets_cli.py` (argparse + rich CLI). Namespaces (`DocumentsNamespace`, `RunsNamespace`, `EvalsNamespace`) are injected into `ConduitDatasetAsync` at construction; `ConduitDatasetSync` wraps them via a `_SyncProxy` that calls `loop.run_until_complete()`.

**Tech Stack:** asyncpg (pool), rich (CLI output), argparse (CLI parsing), pytest-asyncio (tests, `asyncio_mode = "auto"` already configured)

**Design doc:** `docs/plans/2026-03-16-conduit-dataset-design.md` — source of truth for all behaviour, exceptions, and failure modes.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `evals/persist.py` | Modify | Updated DDL: `configs` table, `project` column on `run_results`/`eval_results` |
| `evals/dataset.py` | Create | Exceptions, `_SyncProxy`, namespace classes, `ConduitDatasetAsync`, `ConduitDatasetSync`, `ConduitDataset` alias |
| `evals/tests/__init__.py` | Create | Empty — marks directory as package |
| `evals/tests/conftest.py` | Create | Session pool fixture, per-test project name + cleanup |
| `evals/tests/test_dataset.py` | Create | All 18 acceptance criteria tests |
| `src/conduit/apps/scripts/datasets_cli.py` | Create | `argparse` + `rich` CLI: `status`, `inspect`, `inspect --scores` |
| `pyproject.toml` | Modify | Register `conduit-dataset` entrypoint |

---

## Task 1: Migrate DDL in persist.py

**Files:**
- Modify: `evals/persist.py`

> Adds `configs` KV table. Adds `project TEXT NOT NULL` to `run_results` (now part of PK) and `eval_results`. The old schema had no `project` column. Drop and recreate tables before running `ensure_tables()` if they already exist.

- [ ] **Step 1.1: Drop existing tables in the `evals` database**

Connect to postgres and run:
```sql
DROP TABLE IF EXISTS eval_results CASCADE;
DROP TABLE IF EXISTS run_results CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS configs CASCADE;
```

- [ ] **Step 1.2: Replace `_DDL` in `evals/persist.py`**

Replace the entire `_DDL` string with:
```python
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
```

Also remove the old `save_corpus`, `save_run_results`, `save_eval_results`, `load_corpus` functions from `persist.py` — they are superseded by `dataset.py`. Keep only `_pool`, `_get_pool`, `close_pool`, `ensure_tables`, `_doc_hash`, `_dump`.

- [ ] **Step 1.3: Verify `ensure_tables()` runs cleanly**
```bash
cd /home/fishhouses/Brian_Code/conduit-project/evals
python -c "import asyncio; from persist import ensure_tables; asyncio.run(ensure_tables()); print('OK')"
```
Expected output: `OK`

- [ ] **Step 1.4: Commit**
```bash
git add evals/persist.py
git commit -m "feat: update persist.py DDL — configs table, project column on run/eval results"
```

---

## Task 2: Test Infrastructure + Exceptions

**Files:**
- Create: `evals/tests/__init__.py`
- Create: `evals/tests/conftest.py`
- Create: `evals/dataset.py` (exceptions + skeleton only)

- [ ] **Step 2.1: Create `evals/tests/__init__.py`**
```python
# intentionally empty
```

- [ ] **Step 2.2: Create `evals/tests/conftest.py`**
```python
import asyncio
import uuid
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "dbclients-project" / "src"))

from persist import ensure_tables, _get_pool


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def pool():
    p = await _get_pool()
    await ensure_tables()
    return p


@pytest.fixture
def project():
    """Unique project name per test — prevents cross-test pollution."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def cleanup(pool, project):
    yield
    # Cleanup order: run_results first (evals cascade via FK), then documents
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM run_results WHERE project = $1", project)
        await conn.execute("DELETE FROM documents WHERE project = $1", project)


@pytest_asyncio.fixture
async def cd(project, cleanup, pool):
    from dataset import ConduitDatasetAsync
    return ConduitDatasetAsync(project, pool=pool)
```

- [ ] **Step 2.3: Create `evals/dataset.py` with exceptions and empty skeletons**
```python
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dbclients-project" / "src"))

from evals import RunInput, RunResult, EvalResult, RunOutput

logger = logging.getLogger("conduit.dataset")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DocumentsHaveRunsError(Exception):
    """Raised when delete_all() or swap() is called but run_results exist."""


class GoldStandardExistsError(Exception):
    """Raised when save_gold_standards() would overwrite an existing reference."""


class DocumentNotFoundError(Exception):
    """Raised when save_gold_standards() references a source_id not in documents."""


class BatchSaveError(Exception):
    """Raised by evals.save() when one or more per-result transactions fail."""
    def __init__(self, failures: list[tuple[EvalResult, Exception]]):
        self.failures = failures
        super().__init__(f"{len(failures)} eval result(s) failed to save")


# ---------------------------------------------------------------------------
# Pool + helpers
# ---------------------------------------------------------------------------

def _compute_config_id(config: dict) -> str:
    return hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8]


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _get_pool_default():
    from persist import _get_pool
    return await _get_pool()


# ---------------------------------------------------------------------------
# Namespace skeletons (implementations added per task)
# ---------------------------------------------------------------------------

class DocumentsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn


class RunsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn


class EvalsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn


# ---------------------------------------------------------------------------
# ConduitDatasetAsync
# ---------------------------------------------------------------------------

class ConduitDatasetAsync:
    def __init__(self, project: str, pool=None):
        self._project = project
        self._pool = pool
        pool_fn = (lambda p: (lambda: _get_pool_default()))(pool) \
            if pool is None else (lambda p: (lambda: asyncio.coroutine(lambda: p)()))(pool)
        # Cleaner pool_fn construction:
        _p = pool
        async def pool_fn():
            if _p is not None:
                return _p
            return await _get_pool_default()

        self.documents = DocumentsNamespace(project, pool_fn)
        self.runs = RunsNamespace(project, pool_fn)
        self.evals = EvalsNamespace(project, pool_fn)

    @classmethod
    async def list_projects(cls, pool=None) -> list[str]:
        if pool is None:
            pool = await _get_pool_default()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT project FROM documents ORDER BY project"
            )
        return [r["project"] for r in rows]


# ---------------------------------------------------------------------------
# Sync proxy + ConduitDatasetSync
# ---------------------------------------------------------------------------

class _SyncProxy:
    """Wraps an async namespace object, making every coroutine method blocking."""

    def __init__(self, async_ns, loop: asyncio.AbstractEventLoop, closed_fn: Callable):
        self._ns = async_ns
        self._loop = loop
        self._closed_fn = closed_fn

    def __getattr__(self, name: str):
        if self._closed_fn():
            raise RuntimeError("ConduitDataset is closed")
        attr = getattr(self._ns, name)
        if asyncio.iscoroutinefunction(attr):
            def wrapper(*args, **kwargs):
                if self._closed_fn():
                    raise RuntimeError("ConduitDataset is closed")
                return self._loop.run_until_complete(attr(*args, **kwargs))
            return wrapper
        return attr


class ConduitDatasetSync:
    """
    Sync wrapper around ConduitDatasetAsync.
    Creates a new event loop at __init__. Raises RuntimeError if a loop is
    already running (i.e. called from async context — use ConduitDatasetAsync).
    Supports use as a context manager.
    """

    def __init__(self, project: str, pool=None):
        try:
            running = asyncio.get_event_loop().is_running()
        except RuntimeError:
            running = False
        if running:
            raise RuntimeError("Use ConduitDatasetAsync in async contexts")

        self._loop = asyncio.new_event_loop()
        self._closed = False
        self._async = ConduitDatasetAsync(project, pool=pool)

        def is_closed():
            return self._closed

        self.documents = _SyncProxy(self._async.documents, self._loop, is_closed)
        self.runs = _SyncProxy(self._async.runs, self._loop, is_closed)
        self.evals = _SyncProxy(self._async.evals, self._loop, is_closed)

    def close(self) -> None:
        if not self._closed:
            self._loop.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def list_projects(cls) -> list[str]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(ConduitDatasetAsync.list_projects())
        finally:
            loop.close()


ConduitDataset = ConduitDatasetSync
```

- [ ] **Step 2.4: Verify skeleton imports without error**
```bash
cd /home/fishhouses/Brian_Code/conduit-project/evals
python -c "from dataset import ConduitDataset, ConduitDatasetAsync, BatchSaveError; print('OK')"
```
Expected: `OK`

- [ ] **Step 2.5: Commit**
```bash
git add evals/tests/__init__.py evals/tests/conftest.py evals/dataset.py
git commit -m "feat: test infrastructure, exceptions, dataset skeletons"
```

---

## Task 3: DocumentsNamespace — list() and save()

**Fulfills:** AC 1, AC 4, AC 17 (documents)

**Files:**
- Modify: `evals/dataset.py`
- Create: `evals/tests/test_dataset.py`

- [ ] **Step 3.1: Write failing test — AC 1**

Create `evals/tests/test_dataset.py`:
```python
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import RunInput, RunResult, EvalResult, RunOutput
from dataset import (
    ConduitDatasetAsync,
    ConduitDataset,
    DocumentsHaveRunsError,
    GoldStandardExistsError,
    DocumentNotFoundError,
    BatchSaveError,
)


# ── AC 1 ─────────────────────────────────────────────────────────────────────
# Given 3 documents saved (two with reference, one without), list() returns
# exactly 3 RunInputs with reference populated correctly.

async def test_documents_list_returns_run_inputs_with_reference(cd):
    """AC 1"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text one", reference="ref one"),
        RunInput(source_id="doc2", data="text two", reference="ref two"),
        RunInput(source_id="doc3", data="text three"),
    ])
    result = await cd.documents.list()
    assert len(result) == 3
    by_id = {r.source_id: r for r in result}
    assert by_id["doc1"].reference == "ref one"
    assert by_id["doc2"].reference == "ref two"
    assert by_id["doc3"].reference is None
    assert by_id["doc1"].data == "text one"
```

- [ ] **Step 3.2: Run test to verify it fails**
```bash
cd /home/fishhouses/Brian_Code/conduit-project/evals
python -m pytest tests/test_dataset.py::test_documents_list_returns_run_inputs_with_reference -v
```
Expected: FAIL — `AttributeError` (no `list` or `save` method yet)

- [ ] **Step 3.3: Implement `DocumentsNamespace.list()` and `save()`**

Replace the `DocumentsNamespace` skeleton in `dataset.py` with:
```python
class DocumentsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn

    async def list(self) -> list[RunInput]:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_id, text, reference, metadata
                FROM   documents
                WHERE  project = $1
                ORDER  BY source_id
                """,
                self._project,
            )
        return [
            RunInput(
                source_id=r["source_id"],
                data=r["text"],
                reference=r["reference"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            )
            for r in rows
        ]

    async def save(self, items: list[RunInput]) -> None:
        if not items:
            return
        pool = await self._pool_fn()
        rows = [
            (
                _doc_hash(item.data),
                self._project,
                item.source_id,
                item.data,
                item.reference,
                json.dumps(item.metadata) if item.metadata else None,
            )
            for item in items
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO documents
                    (doc_hash, project, source_id, text, reference, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (project, source_id) DO UPDATE SET
                    doc_hash = EXCLUDED.doc_hash,
                    text     = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )
        logger.debug("documents.save project=%s count=%d", self._project, len(items))

    # needs_gold_standard, save_gold_standards, swap, delete_all added in later tasks
```

- [ ] **Step 3.4: Run AC 1 test to verify it passes**
```bash
python -m pytest tests/test_dataset.py::test_documents_list_returns_run_inputs_with_reference -v
```
Expected: PASS

- [ ] **Step 3.5: Write failing test — AC 4**
```python
# ── AC 4 ─────────────────────────────────────────────────────────────────────
# save() does NOT overwrite an existing non-null reference on conflict.

async def test_documents_save_does_not_overwrite_reference(cd):
    """AC 4"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text", reference="keep this")
    ])
    # Second save with reference=None must not clear the stored reference
    await cd.documents.save([
        RunInput(source_id="doc1", data="text updated")
    ])
    result = await cd.documents.list()
    assert result[0].reference == "keep this"
```

- [ ] **Step 3.6: Run to verify it passes (ON CONFLICT excludes reference column)**
```bash
python -m pytest tests/test_dataset.py::test_documents_save_does_not_overwrite_reference -v
```
Expected: PASS — the `ON CONFLICT DO UPDATE` clause intentionally omits `reference`; no code change needed. If FAIL, the `ON CONFLICT` clause must be verified to not include `reference`.

- [ ] **Step 3.7: Write failing test — AC 17 (documents)**
```python
# ── AC 17 (documents) ────────────────────────────────────────────────────────
# Saving the same document twice produces exactly one row.

async def test_documents_save_is_idempotent(cd, pool, project):
    """AC 17 — documents"""
    item = RunInput(source_id="doc1", data="text")
    await cd.documents.save([item])
    await cd.documents.save([item])
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1 AND source_id = $2",
            project, "doc1",
        )
    assert count == 1
```

- [ ] **Step 3.8: Run to verify it passes**
```bash
python -m pytest tests/test_dataset.py::test_documents_save_is_idempotent -v
```
Expected: PASS

- [ ] **Step 3.9: Commit**
```bash
git add evals/dataset.py evals/tests/test_dataset.py
git commit -m "feat: DocumentsNamespace list() and save() — AC 1, AC 4, AC 17 (documents)"
```

---

## Task 4: DocumentsNamespace — swap() and delete_all()

**Fulfills:** AC 2, AC 3

**Files:**
- Modify: `evals/dataset.py`
- Modify: `evals/tests/test_dataset.py`

- [ ] **Step 4.1: Write failing test — AC 2**
```python
# ── AC 2 ─────────────────────────────────────────────────────────────────────
# swap() is atomic: a failed INSERT leaves the original documents unchanged.
# We trigger a NOT NULL violation (data=None) mid-insert to force rollback.

async def test_documents_swap_is_atomic(cd):
    """AC 2"""
    await cd.documents.save([RunInput(source_id="original", data="original text")])
    with pytest.raises(Exception):
        await cd.documents.swap([
            RunInput(source_id="good", data="good text"),
            RunInput(source_id="bad", data=None),  # NOT NULL on text column → DB error
        ])
    remaining = await cd.documents.list()
    assert len(remaining) == 1
    assert remaining[0].source_id == "original"
```

- [ ] **Step 4.2: Run to verify it fails**
```bash
python -m pytest tests/test_dataset.py::test_documents_swap_is_atomic -v
```
Expected: FAIL — `AttributeError: 'DocumentsNamespace' object has no attribute 'swap'`

- [ ] **Step 4.3: Write failing test — AC 3**
```python
# ── AC 3 ─────────────────────────────────────────────────────────────────────
# delete_all() raises DocumentsHaveRunsError if runs exist and force=False.
# delete_all(force=True) leaves zero documents, runs, and evals.

async def test_documents_delete_all_raises_if_runs_exist(cd, pool, project):
    """AC 3 — guard"""
    await cd.documents.save([RunInput(source_id="doc1", data="text")])
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO configs (config_id, config) VALUES ('aaaaaaaa', '{}'::jsonb)"
            " ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            """
            INSERT INTO run_results (project, strategy, config_id, source_id, output)
            VALUES ($1, 'Strat', 'aaaaaaaa', 'doc1', 'out')
            ON CONFLICT DO NOTHING
            """,
            project,
        )
    with pytest.raises(DocumentsHaveRunsError):
        await cd.documents.delete_all()


async def test_documents_delete_all_force_clears_all(cd, pool, project):
    """AC 3 — force=True"""
    await cd.documents.save([RunInput(source_id="doc1", data="text")])
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO configs (config_id, config) VALUES ('aaaaaaaa', '{}'::jsonb)"
            " ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            """
            INSERT INTO run_results (project, strategy, config_id, source_id, output)
            VALUES ($1, 'Strat', 'aaaaaaaa', 'doc1', 'out')
            ON CONFLICT DO NOTHING
            """,
            project,
        )
    await cd.documents.delete_all(force=True)
    async with pool.acquire() as conn:
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1", project
        )
        runs = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project = $1", project
        )
        evals = await conn.fetchval(
            "SELECT COUNT(*) FROM eval_results WHERE project = $1", project
        )
    assert docs == 0
    assert runs == 0
    assert evals == 0
```

- [ ] **Step 4.3a: Run AC 3 tests to verify they fail**
```bash
python -m pytest \
    tests/test_dataset.py::test_documents_delete_all_raises_if_runs_exist \
    tests/test_dataset.py::test_documents_delete_all_force_clears_all -v
```
Expected: FAIL — `AttributeError: 'DocumentsNamespace' object has no attribute 'delete_all'`

- [ ] **Step 4.4: Implement `swap()` and `delete_all()` in `DocumentsNamespace`**

Add to the `DocumentsNamespace` class:
```python
async def swap(self, items: list[RunInput], force: bool = False) -> None:
    pool = await self._pool_fn()
    async with pool.acquire() as conn:
        async with conn.transaction():
            run_count = await conn.fetchval(
                "SELECT COUNT(*) FROM run_results WHERE project = $1",
                self._project,
            )
            if run_count > 0 and not force:
                raise DocumentsHaveRunsError(
                    f"Project '{self._project}' has {run_count} run(s). "
                    "Call runs.delete_all() first or pass force=True."
                )
            if force:
                await conn.execute(
                    "DELETE FROM run_results WHERE project = $1", self._project
                )
            await conn.execute(
                "DELETE FROM documents WHERE project = $1", self._project
            )
            if items:
                rows = [
                    (
                        _doc_hash(item.data),
                        self._project,
                        item.source_id,
                        item.data,
                        item.reference,
                        json.dumps(item.metadata) if item.metadata else None,
                    )
                    for item in items
                ]
                await conn.executemany(
                    """
                    INSERT INTO documents
                        (doc_hash, project, source_id, text, reference, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    rows,
                )
    logger.info("documents.swap project=%s count=%d", self._project, len(items))

async def delete_all(self, force: bool = False) -> None:
    pool = await self._pool_fn()
    async with pool.acquire() as conn:
        run_count = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project = $1", self._project
        )
        if run_count > 0 and not force:
            raise DocumentsHaveRunsError(
                f"Project '{self._project}' has {run_count} run(s). "
                "Call runs.delete_all() first or pass force=True."
            )
        if force:
            # eval_results cascade automatically via FK on run_results
            await conn.execute(
                "DELETE FROM run_results WHERE project = $1", self._project
            )
        await conn.execute(
            "DELETE FROM documents WHERE project = $1", self._project
        )
    logger.warning("documents.delete_all project=%s force=%s", self._project, force)
```

- [ ] **Step 4.5: Run AC 2 and AC 3 tests to verify they pass**
```bash
python -m pytest tests/test_dataset.py::test_documents_swap_is_atomic \
    tests/test_dataset.py::test_documents_delete_all_raises_if_runs_exist \
    tests/test_dataset.py::test_documents_delete_all_force_clears_all -v
```
Expected: all PASS

- [ ] **Step 4.6: Commit**
```bash
git add evals/dataset.py evals/tests/test_dataset.py
git commit -m "feat: DocumentsNamespace swap() and delete_all() — AC 2, AC 3"
```

---

## Task 5: DocumentsNamespace — needs_gold_standard() and save_gold_standards()

**Fulfills:** AC 5, AC 6, AC 7

**Files:**
- Modify: `evals/dataset.py`
- Modify: `evals/tests/test_dataset.py`

- [ ] **Step 5.1: Write failing test — AC 5**
```python
# ── AC 5 ─────────────────────────────────────────────────────────────────────
# save_gold_standards() raises GoldStandardExistsError when any item already
# has a reference and force=False. No rows are written when the error is raised.

async def test_save_gold_standards_raises_if_reference_exists(cd):
    """AC 5 — pre-validate, no partial writes"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text 1", reference="existing"),
        RunInput(source_id="doc2", data="text 2"),
    ])
    with pytest.raises(GoldStandardExistsError):
        await cd.documents.save_gold_standards([
            RunInput(source_id="doc1", data="text 1", reference="new gold"),
            RunInput(source_id="doc2", data="text 2", reference="new gold 2"),
        ])
    # doc2 must NOT have been written (pre-validate, no partial writes)
    result = await cd.documents.list()
    by_id = {r.source_id: r for r in result}
    assert by_id["doc2"].reference is None


async def test_save_gold_standards_force_overwrites(cd):
    """AC 5 — force=True"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text", reference="old ref")
    ])
    await cd.documents.save_gold_standards(
        [RunInput(source_id="doc1", data="text", reference="new ref")],
        force=True,
    )
    result = await cd.documents.list()
    assert result[0].reference == "new ref"
```

- [ ] **Step 5.2: Run to verify it fails**
```bash
python -m pytest tests/test_dataset.py::test_save_gold_standards_raises_if_reference_exists -v
```
Expected: FAIL — `AttributeError`

- [ ] **Step 5.3: Write failing test — AC 6**
```python
# ── AC 6 ─────────────────────────────────────────────────────────────────────
# save_gold_standards() raises DocumentNotFoundError when any source_id is
# missing from the project. No rows are written.

async def test_save_gold_standards_raises_if_source_id_missing(cd):
    """AC 6"""
    await cd.documents.save([RunInput(source_id="doc1", data="text")])
    with pytest.raises(DocumentNotFoundError):
        await cd.documents.save_gold_standards([
            RunInput(source_id="doc1", data="text", reference="gold 1"),
            RunInput(source_id="missing", data="x", reference="gold missing"),
        ])
    # doc1 must NOT have been written (pre-validate, no partial writes)
    result = await cd.documents.list()
    assert result[0].reference is None
```

- [ ] **Step 5.4: Write failing test — AC 7**
```python
# ── AC 7 ─────────────────────────────────────────────────────────────────────
# Items where RunInput.reference is None are silently skipped.

async def test_save_gold_standards_skips_none_reference(cd):
    """AC 7"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text 1"),
        RunInput(source_id="doc2", data="text 2"),
    ])
    await cd.documents.save_gold_standards([
        RunInput(source_id="doc1", data="text 1", reference="gold 1"),
        RunInput(source_id="doc2", data="text 2", reference=None),  # skipped
    ])
    result = await cd.documents.list()
    by_id = {r.source_id: r for r in result}
    assert by_id["doc1"].reference == "gold 1"
    assert by_id["doc2"].reference is None
```

- [ ] **Step 5.5: Implement `needs_gold_standard()` and `save_gold_standards()`**

Add to `DocumentsNamespace`:
```python
async def needs_gold_standard(self) -> list[RunInput]:
    pool = await self._pool_fn()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_id, text, reference, metadata
            FROM   documents
            WHERE  project = $1 AND reference IS NULL
            ORDER  BY source_id
            """,
            self._project,
        )
    return [
        RunInput(
            source_id=r["source_id"],
            data=r["text"],
            reference=None,
            metadata=json.loads(r["metadata"]) if r["metadata"] else None,
        )
        for r in rows
    ]

async def save_gold_standards(
    self, items: list[RunInput], force: bool = False
) -> None:
    # Filter out items with reference=None (silent no-op per spec)
    to_write = [i for i in items if i.reference is not None]
    if not to_write:
        return

    pool = await self._pool_fn()
    async with pool.acquire() as conn:
        # Pre-validate: all source_ids must exist in this project
        source_ids = [i.source_id for i in to_write]
        existing_rows = await conn.fetch(
            """
            SELECT source_id, reference
            FROM   documents
            WHERE  project = $1 AND source_id = ANY($2::text[])
            """,
            self._project,
            source_ids,
        )
        existing = {r["source_id"]: r["reference"] for r in existing_rows}

        missing = [sid for sid in source_ids if sid not in existing]
        if missing:
            raise DocumentNotFoundError(
                f"source_id(s) not found in project '{self._project}': {missing}"
            )

        conflicts = [sid for sid, ref in existing.items() if ref is not None]
        if conflicts:
            if not force:
                logger.warning(
                    "save_gold_standards: existing references found project=%s ids=%s",
                    self._project, conflicts,
                )
                raise GoldStandardExistsError(
                    f"Gold standards already exist for: {conflicts}. "
                    "Pass force=True to overwrite."
                )
            else:
                logger.warning(
                    "save_gold_standards: force=True, overwriting existing references "
                    "project=%s ids=%s",
                    self._project, conflicts,
                )

        # All validation passed — write
        rows = [(i.reference, self._project, i.source_id) for i in to_write]
        await conn.executemany(
            """
            UPDATE documents SET reference = $1
            WHERE  project = $2 AND source_id = $3
            """,
            rows,
        )
```

- [ ] **Step 5.6: Run AC 5, AC 6, AC 7 tests**
```bash
python -m pytest \
    tests/test_dataset.py::test_save_gold_standards_raises_if_reference_exists \
    tests/test_dataset.py::test_save_gold_standards_force_overwrites \
    tests/test_dataset.py::test_save_gold_standards_raises_if_source_id_missing \
    tests/test_dataset.py::test_save_gold_standards_skips_none_reference -v
```
Expected: all PASS

- [ ] **Step 5.7: Commit**
```bash
git add evals/dataset.py evals/tests/test_dataset.py
git commit -m "feat: needs_gold_standard() and save_gold_standards() — AC 5, AC 6, AC 7"
```

---

## Task 6: RunsNamespace — save() and configs deduplication

**Fulfills:** AC 13, AC 17 (runs)

**Files:**
- Modify: `evals/dataset.py`
- Modify: `evals/tests/test_dataset.py`

- [ ] **Step 6.1: Write failing test — AC 13**
```python
# ── AC 13 ────────────────────────────────────────────────────────────────────
# configs table has exactly one row per unique config dict regardless of how
# many runs share that config.

async def test_runs_save_deduplicates_configs(cd, pool, project):
    """AC 13"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)  # import from dataset
    results = [
        RunResult(
            strategy="OneShotSummarizer",
            config_id=config_id,
            source_id=f"doc{i}",
            config=config,
            output=RunOutput(output=f"summary {i}", metadata={}),
        )
        for i in range(3)
    ]
    await cd.runs.save(results)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM configs WHERE config_id = $1", config_id
        )
    assert count == 1
```

Add this import at the top of the test file:
```python
from dataset import _compute_config_id
```

- [ ] **Step 6.2: Run to verify it fails**
```bash
python -m pytest tests/test_dataset.py::test_runs_save_deduplicates_configs -v
```
Expected: FAIL — `AttributeError: 'RunsNamespace' object has no attribute 'save'`

- [ ] **Step 6.3: Write failing test — AC 17 (runs)**
```python
# ── AC 17 (runs) ─────────────────────────────────────────────────────────────
# Saving the same RunResult twice produces exactly one row.

async def test_runs_save_is_idempotent(cd, pool, project):
    """AC 17 — runs"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)
    result = RunResult(
        strategy="OneShotSummarizer",
        config_id=config_id,
        source_id="doc1",
        config=config,
        output=RunOutput(output="summary", metadata={}),
    )
    await cd.runs.save([result])
    await cd.runs.save([result])
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM run_results
            WHERE project = $1 AND strategy = $2 AND config_id = $3 AND source_id = $4
            """,
            project, "OneShotSummarizer", config_id, "doc1",
        )
    assert count == 1
```

- [ ] **Step 6.4: Implement `RunsNamespace.save()`**

Replace the `RunsNamespace` skeleton with:
```python
class RunsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn

    async def save(self, results: list[RunResult]) -> None:
        if not results:
            return
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            for r in results:
                config = r.config if isinstance(r.config, dict) else r.config.model_dump()
                config_id = _compute_config_id(config)
                # 1. Upsert config
                await conn.execute(
                    """
                    INSERT INTO configs (config_id, config)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (config_id) DO NOTHING
                    """,
                    config_id,
                    json.dumps(config),
                )
                # 2. Upsert run_result
                await conn.execute(
                    """
                    INSERT INTO run_results
                        (project, strategy, config_id, source_id, reference_id,
                         output, output_metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    ON CONFLICT (project, strategy, config_id, source_id) DO UPDATE SET
                        reference_id    = EXCLUDED.reference_id,
                        output          = EXCLUDED.output,
                        output_metadata = EXCLUDED.output_metadata,
                        created_at      = now()
                    """,
                    self._project,
                    r.strategy,
                    config_id,
                    r.source_id,
                    r.reference_id,
                    r.output.output,
                    json.dumps(r.output.metadata) if r.output.metadata else None,
                )
        logger.debug("runs.save project=%s count=%d", self._project, len(results))

    async def delete_all(self) -> None:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM run_results WHERE project = $1", self._project
            )
        logger.debug("runs.delete_all project=%s", self._project)

    async def list(self, strategy=None, config_id=None, source_id=None, config=None):
        # Implemented in Task 7
        raise NotImplementedError

    async def list_configs(self):
        # Implemented in Task 7
        raise NotImplementedError
```

- [ ] **Step 6.5: Run AC 13 and AC 17 (runs) tests**
```bash
python -m pytest \
    tests/test_dataset.py::test_runs_save_deduplicates_configs \
    tests/test_dataset.py::test_runs_save_is_idempotent -v
```
Expected: all PASS

- [ ] **Step 6.6: Commit**
```bash
git add evals/dataset.py evals/tests/test_dataset.py
git commit -m "feat: RunsNamespace save() and delete_all() — AC 13, AC 17 (runs)"
```

---

## Task 7: RunsNamespace — list() and list_configs()

**Fulfills:** AC 8, AC 9, AC 10, AC 11

**Files:**
- Modify: `evals/dataset.py`
- Modify: `evals/tests/test_dataset.py`

- [ ] **Step 7.1: Write failing test — AC 8**
```python
# ── AC 8 ─────────────────────────────────────────────────────────────────────
# runs.list(config={...}) returns only runs whose config_id matches the hash
# of that dict.

async def test_runs_list_filters_by_config_dict(cd, project):
    """AC 8"""
    config_a = {"model": "gpt-oss:latest"}
    config_b = {"model": "deepseek:latest"}
    id_a = _compute_config_id(config_a)
    id_b = _compute_config_id(config_b)

    await cd.runs.save([
        RunResult(strategy="S", config_id=id_a, source_id="doc1",
                  config=config_a, output=RunOutput(output="out", metadata={})),
        RunResult(strategy="S", config_id=id_b, source_id="doc1",
                  config=config_b, output=RunOutput(output="out", metadata={})),
    ])
    results = await cd.runs.list(config=config_a)
    assert len(results) == 1
    assert results[0].config_id == id_a
```

- [ ] **Step 7.2: Run to verify it fails**
```bash
python -m pytest tests/test_dataset.py::test_runs_list_filters_by_config_dict -v
```
Expected: FAIL — `NotImplementedError`

- [ ] **Step 7.3: Write failing test — AC 9**
```python
# ── AC 9 ─────────────────────────────────────────────────────────────────────
# runs.list(config=unknown_dict) returns [] — not an error.

async def test_runs_list_unknown_config_returns_empty(cd):
    """AC 9"""
    result = await cd.runs.list(config={"model": "never-saved"})
    assert result == []
```

- [ ] **Step 7.4: Write failing test — AC 10**
```python
# ── AC 10 ────────────────────────────────────────────────────────────────────
# runs.list(config=..., config_id=...) raises ValueError at call site.

async def test_runs_list_raises_on_conflicting_config_args(cd):
    """AC 10"""
    with pytest.raises(ValueError, match="mutually exclusive"):
        await cd.runs.list(config={"model": "x"}, config_id="abc12345")
```

- [ ] **Step 7.5: Write failing test — AC 11**
```python
# ── AC 11 ────────────────────────────────────────────────────────────────────
# list_configs() returns distinct config dicts ordered by configs.created_at ASC.

async def test_runs_list_configs_ordered_by_created_at(cd):
    """AC 11"""
    config_a = {"model": "first"}
    config_b = {"model": "second"}
    id_a = _compute_config_id(config_a)
    id_b = _compute_config_id(config_b)

    await cd.runs.save([
        RunResult(strategy="S", config_id=id_a, source_id="doc1",
                  config=config_a, output=RunOutput(output="o", metadata={})),
    ])
    await cd.runs.save([
        RunResult(strategy="S", config_id=id_b, source_id="doc2",
                  config=config_b, output=RunOutput(output="o", metadata={})),
    ])
    configs = await cd.runs.list_configs()
    assert len(configs) == 2
    assert configs[0] == config_a  # first saved comes first
    assert configs[1] == config_b
    # No duplicates
    assert len(configs) == len({json.dumps(c, sort_keys=True) for c in configs})
```

- [ ] **Step 7.6: Implement `RunsNamespace.list()` and `list_configs()`**

Replace the `list` and `list_configs` stubs in `RunsNamespace`:
```python
async def list(
    self,
    strategy: str | None = None,
    config_id: str | None = None,
    source_id: str | None = None,
    config: dict | None = None,
) -> list[RunResult]:
    if config is not None and config_id is not None:
        raise ValueError(
            "config and config_id are mutually exclusive — provide one or neither"
        )
    if config is not None:
        config_id = _compute_config_id(config)

    pool = await self._pool_fn()
    async with pool.acquire() as conn:
        conditions = ["r.project = $1"]
        params = [self._project]
        idx = 2
        if strategy is not None:
            conditions.append(f"r.strategy = ${idx}")
            params.append(strategy)
            idx += 1
        if config_id is not None:
            conditions.append(f"r.config_id = ${idx}")
            params.append(config_id)
            idx += 1
        if source_id is not None:
            conditions.append(f"r.source_id = ${idx}")
            params.append(source_id)
            idx += 1

        where = " AND ".join(conditions)
        rows = await conn.fetch(
            f"""
            SELECT r.strategy, r.config_id, r.source_id, r.reference_id,
                   r.output, r.output_metadata, c.config
            FROM   run_results r
            JOIN   configs c USING (config_id)
            WHERE  {where}
            """,
            *params,
        )
    return [
        RunResult(
            strategy=r["strategy"],
            config_id=r["config_id"],
            source_id=r["source_id"],
            reference_id=r["reference_id"],
            config=json.loads(r["config"]),
            output=RunOutput(
                output=r["output"],
                metadata=json.loads(r["output_metadata"]) if r["output_metadata"] else {},
            ),
        )
        for r in rows
    ]

async def list_configs(self) -> list[dict]:
    pool = await self._pool_fn()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT c.config, c.created_at
            FROM   run_results r
            JOIN   configs c USING (config_id)
            WHERE  r.project = $1
            ORDER  BY c.created_at ASC
            """,
            self._project,
        )
    return [json.loads(r["config"]) for r in rows]
```

- [ ] **Step 7.7: Run AC 8, AC 9, AC 10, AC 11 tests**
```bash
python -m pytest \
    tests/test_dataset.py::test_runs_list_filters_by_config_dict \
    tests/test_dataset.py::test_runs_list_unknown_config_returns_empty \
    tests/test_dataset.py::test_runs_list_raises_on_conflicting_config_args \
    tests/test_dataset.py::test_runs_list_configs_ordered_by_created_at -v
```
Expected: all PASS

- [ ] **Step 7.8: Commit**
```bash
git add evals/dataset.py evals/tests/test_dataset.py
git commit -m "feat: RunsNamespace list() and list_configs() — AC 8, AC 9, AC 10, AC 11"
```

---

## Task 8: EvalsNamespace — save() with per-result transactions

**Fulfills:** AC 12, AC 17 (evals)

**Files:**
- Modify: `evals/dataset.py`
- Modify: `evals/tests/test_dataset.py`

- [ ] **Step 8.1: Write failing test — AC 12**
```python
# ── AC 12 ────────────────────────────────────────────────────────────────────
# evals.save(): one result's transaction failure rolls back both its RunResult
# and eval write; the rest of the batch succeeds; BatchSaveError is raised.

async def test_evals_save_per_result_transaction_and_batch_error(cd, pool, project):
    """AC 12"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)

    good = EvalResult(
        run_result=RunResult(
            strategy="S", config_id=config_id, source_id="doc1",
            config=config, output=RunOutput(output="out", metadata={}),
        ),
        score=0.9,
    )
    # Bad: config_id is a hash that doesn't exist in configs and can't be
    # auto-inserted because we provide a mismatched config dict that computes
    # to a *different* config_id — forcing an FK violation on run_results.
    bad_run = RunResult(
        strategy="S",
        config_id="deadbeef",        # valid-looking but won't match computed hash
        source_id="doc2",
        config={"model": "other"},   # _compute_config_id(this) != "deadbeef"
        output=RunOutput(output="out", metadata={}),
    )
    bad = EvalResult(run_result=bad_run, score=0.5)

    with pytest.raises(BatchSaveError) as exc_info:
        await cd.evals.save([good, bad], eval_function="test_scorer")

    assert len(exc_info.value.failures) == 1
    assert exc_info.value.failures[0][0] is bad

    # good result must be persisted; bad must not
    async with pool.acquire() as conn:
        good_run = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project=$1 AND source_id='doc1'",
            project,
        )
        bad_run_count = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project=$1 AND source_id='doc2'",
            project,
        )
    assert good_run == 1
    assert bad_run_count == 0
```

- [ ] **Step 8.2: Run to verify it fails**
```bash
python -m pytest tests/test_dataset.py::test_evals_save_per_result_transaction_and_batch_error -v
```
Expected: FAIL — `AttributeError: 'EvalsNamespace' object has no attribute 'save'`

- [ ] **Step 8.3: Write failing test — AC 17 (evals)**
```python
# ── AC 17 (evals) ────────────────────────────────────────────────────────────
# Saving the same EvalResult twice produces exactly one row.

async def test_evals_save_is_idempotent(cd, pool, project):
    """AC 17 — evals"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)
    er = EvalResult(
        run_result=RunResult(
            strategy="S", config_id=config_id, source_id="doc1",
            config=config, output=RunOutput(output="out", metadata={}),
        ),
        score=0.8,
    )
    await cd.evals.save([er], eval_function="scorer")
    await cd.evals.save([er], eval_function="scorer")
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM eval_results
            WHERE project=$1 AND eval_function='scorer'
            """,
            project,
        )
    assert count == 1
```

- [ ] **Step 8.4: Implement `EvalsNamespace`**

Replace the `EvalsNamespace` skeleton with:
```python
class EvalsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn

    async def save(
        self, results: list[EvalResult], eval_function: str
    ) -> None:
        if not results:
            return
        pool = await self._pool_fn()
        failures: list[tuple[EvalResult, Exception]] = []

        for er in results:
            r = er.run_result
            config = r.config if isinstance(r.config, dict) else r.config.model_dump()
            config_id = _compute_config_id(config)
            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            """
                            INSERT INTO configs (config_id, config)
                            VALUES ($1, $2::jsonb)
                            ON CONFLICT (config_id) DO NOTHING
                            """,
                            config_id, json.dumps(config),
                        )
                        await conn.execute(
                            """
                            INSERT INTO run_results
                                (project, strategy, config_id, source_id,
                                 reference_id, output, output_metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                            ON CONFLICT (project, strategy, config_id, source_id)
                            DO UPDATE SET
                                output          = EXCLUDED.output,
                                output_metadata = EXCLUDED.output_metadata,
                                created_at      = now()
                            """,
                            self._project, r.strategy, config_id,
                            r.source_id, r.reference_id,
                            r.output.output,
                            json.dumps(r.output.metadata) if r.output.metadata else None,
                        )
                        await conn.execute(
                            """
                            INSERT INTO eval_results
                                (project, strategy, config_id, source_id,
                                 eval_function, score, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                            ON CONFLICT (project, strategy, config_id,
                                         source_id, eval_function)
                            DO UPDATE SET
                                score      = EXCLUDED.score,
                                metadata   = EXCLUDED.metadata,
                                created_at = now()
                            """,
                            self._project, r.strategy, config_id,
                            r.source_id, eval_function, er.score, None,
                        )
            except Exception as exc:
                logger.error(
                    "evals.save transaction failed project=%s strategy=%s "
                    "config_id=%s source_id=%s error=%s",
                    self._project, r.strategy, config_id, r.source_id, exc,
                )
                failures.append((er, exc))

        if failures:
            logger.error(
                "evals.save BatchSaveError project=%s failures=%d attempted=%d",
                self._project, len(failures), len(results),
            )
            raise BatchSaveError(failures)

    async def list(
        self,
        eval_function: str | None = None,
        strategy: str | None = None,
    ) -> list[EvalResult]:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            conditions = ["e.project = $1"]
            params = [self._project]
            idx = 2
            if eval_function is not None:
                conditions.append(f"e.eval_function = ${idx}")
                params.append(eval_function)
                idx += 1
            if strategy is not None:
                conditions.append(f"e.strategy = ${idx}")
                params.append(strategy)
                idx += 1
            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"""
                SELECT e.strategy, e.config_id, e.source_id,
                       e.eval_function, e.score, e.metadata,
                       r.reference_id, r.output, r.output_metadata,
                       c.config
                FROM   eval_results e
                JOIN   run_results r
                       ON  r.project   = e.project
                       AND r.strategy  = e.strategy
                       AND r.config_id = e.config_id
                       AND r.source_id = e.source_id
                JOIN   configs c ON c.config_id = e.config_id
                WHERE  {where}
                """,
                *params,
            )
        return [
            EvalResult(
                run_result=RunResult(
                    strategy=r["strategy"],
                    config_id=r["config_id"],
                    source_id=r["source_id"],
                    reference_id=r["reference_id"],
                    config=json.loads(r["config"]),
                    output=RunOutput(
                        output=r["output"],
                        metadata=json.loads(r["output_metadata"])
                        if r["output_metadata"] else {},
                    ),
                ),
                score=r["score"],
            )
            for r in rows
        ]

    async def delete_all(self) -> None:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM eval_results WHERE project = $1", self._project
            )
```

- [ ] **Step 8.5: Run AC 12 and AC 17 (evals) tests**
```bash
python -m pytest \
    tests/test_dataset.py::test_evals_save_per_result_transaction_and_batch_error \
    tests/test_dataset.py::test_evals_save_is_idempotent -v
```
Expected: all PASS

- [ ] **Step 8.6: Commit**
```bash
git add evals/dataset.py evals/tests/test_dataset.py
git commit -m "feat: EvalsNamespace save() with per-result transactions — AC 12, AC 17 (evals)"
```

---

## Task 9: ConduitDatasetSync — runtime guards and context manager

**Fulfills:** AC 14, AC 15, AC 18

**Files:**
- Modify: `evals/tests/test_dataset.py`

> `ConduitDatasetSync` and `_SyncProxy` are already implemented in Task 2. This task adds the tests.

- [ ] **Step 9.1: Write test — AC 14**

> **Note:** No red phase for AC 14. The running-loop guard was co-located with the `ConduitDatasetSync` skeleton in Task 2 and cannot be separated from it without breaking the skeleton. The guard fires immediately when pytest-asyncio runs the test inside its event loop, so the test goes straight to green.

```python
# ── AC 14 ────────────────────────────────────────────────────────────────────
# ConduitDataset() raises RuntimeError when instantiated inside a running loop.

async def test_sync_raises_when_loop_is_running():
    """AC 14"""
    with pytest.raises(RuntimeError, match="Use ConduitDatasetAsync"):
        ConduitDataset("test_project")
```

- [ ] **Step 9.2: Run to verify it passes (no implementation step needed — guard already in skeleton)**
```bash
python -m pytest tests/test_dataset.py::test_sync_raises_when_loop_is_running -v
```
Expected: PASS — the `asyncio.get_event_loop().is_running()` check fires because pytest-asyncio runs tests inside an event loop. If FAIL: review `ConduitDatasetSync.__init__` detection logic in Task 2 skeleton.

- [ ] **Step 9.3: Write failing test — AC 15**
```python
# ── AC 15 ────────────────────────────────────────────────────────────────────
# Any method call after close() raises RuntimeError("ConduitDataset is closed").

def test_sync_raises_after_close():
    """AC 15"""
    cd = ConduitDataset.__new__(ConduitDataset)
    cd._loop = asyncio.new_event_loop()
    cd._closed = False
    cd._async = ConduitDatasetAsync.__new__(ConduitDatasetAsync)
    cd._async.documents = DocumentsNamespace.__new__(DocumentsNamespace)

    def is_closed():
        return cd._closed

    cd.documents = _SyncProxy(cd._async.documents, cd._loop, is_closed)
    cd.close()

    with pytest.raises(RuntimeError, match="ConduitDataset is closed"):
        cd.documents.list()
```

Add `_SyncProxy` and `DocumentsNamespace` to the imports at the top of the test file:
```python
from dataset import _SyncProxy
```

- [ ] **Step 9.4: Run to verify it passes**
```bash
python -m pytest tests/test_dataset.py::test_sync_raises_after_close -v
```
Expected: PASS. If FAIL: review `_SyncProxy.__getattr__` closed check.

- [ ] **Step 9.5: Write failing test — AC 18**
```python
# ── AC 18 ────────────────────────────────────────────────────────────────────
# ConduitDataset used as context manager calls close() on exit, even if the
# body raises.

def test_sync_context_manager_closes_on_exit():
    """AC 18 — normal exit"""
    # Build a ConduitDataset without triggering the running-loop guard
    # by constructing it outside any async context.
    # This test must be run as a plain (non-async) test function.
    loop = asyncio.new_event_loop()

    class _FakeAsync:
        documents = object()
        runs = object()
        evals = object()

    cd = object.__new__(ConduitDataset)
    cd._loop = loop
    cd._closed = False
    cd._async = _FakeAsync()
    cd.documents = _SyncProxy(cd._async.documents, loop, lambda: cd._closed)
    cd.runs = _SyncProxy(cd._async.runs, loop, lambda: cd._closed)
    cd.evals = _SyncProxy(cd._async.evals, loop, lambda: cd._closed)

    with cd:
        assert not cd._closed

    assert cd._closed
    assert loop.is_closed()


def test_sync_context_manager_closes_on_exception():
    """AC 18 — exception exit"""
    loop = asyncio.new_event_loop()

    class _FakeAsync:
        documents = object()
        runs = object()
        evals = object()

    cd = object.__new__(ConduitDataset)
    cd._loop = loop
    cd._closed = False
    cd._async = _FakeAsync()
    cd.documents = _SyncProxy(cd._async.documents, loop, lambda: cd._closed)
    cd.runs = _SyncProxy(cd._async.runs, loop, lambda: cd._closed)
    cd.evals = _SyncProxy(cd._async.evals, loop, lambda: cd._closed)

    with pytest.raises(ValueError):
        with cd:
            raise ValueError("body error")

    assert cd._closed
```

- [ ] **Step 9.6: Run AC 18 tests**
```bash
python -m pytest \
    tests/test_dataset.py::test_sync_context_manager_closes_on_exit \
    tests/test_dataset.py::test_sync_context_manager_closes_on_exception -v
```
Expected: PASS

- [ ] **Step 9.7: Run all tests to verify no regressions**
```bash
python -m pytest tests/test_dataset.py -v
```
Expected: all PASS

- [ ] **Step 9.8: Commit**
```bash
git add evals/tests/test_dataset.py
git commit -m "test: ConduitDatasetSync guards and context manager — AC 14, AC 15, AC 18"
```

---

## Task 10: CLI — datasets_cli.py

**Fulfills:** AC 16

**Files:**
- Create: `src/conduit/apps/scripts/datasets_cli.py`
- Modify: `evals/tests/test_dataset.py`

The CLI uses `argparse` + `rich`. It calls `ConduitDatasetAsync.list_projects()` and then queries each project's counts. It must exit with code 1 and a clear message when postgres is unreachable.

- [ ] **Step 10.1: Write failing test — AC 16**
```python
# ── AC 16 ────────────────────────────────────────────────────────────────────
# conduit-dataset status exits with code 1 and a message containing the host
# and database name when postgres is unreachable.

def test_cli_status_exits_1_when_db_unreachable():
    """AC 16"""
    # Use a nonexistent host to guarantee connection failure regardless of
    # how _get_pool() constructs its DSN. POSTGRES_PASSWORD would only work
    # if the server rejects auth — a bad host fails at the TCP level, which
    # is reliable and fast.
    import os, subprocess, sys
    env = os.environ.copy()
    env["POSTGRES_HOST"] = "nonexistent-host-conduit-test-xyz.local"
    result = subprocess.run(
        [sys.executable,
         str(Path(__file__).parent.parent.parent /
             "src/conduit/apps/scripts/datasets_cli.py"),
         "status"],
        capture_output=True, text=True, env=env,
        timeout=10,
    )
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Cannot reach postgres" in output
    assert "evals" in output  # database name must appear in the message
```

- [ ] **Step 10.2: Run to verify it fails**
```bash
python -m pytest tests/test_dataset.py::test_cli_status_exits_1_when_db_unreachable -v
```
Expected: FAIL — script doesn't exist yet

- [ ] **Step 10.3: Implement `datasets_cli.py`**

```python
#!/usr/bin/env python3
"""
conduit-dataset CLI — inventory snapshots for ConduitDataset projects.

Commands:
    status                  List all projects with counts per stage.
    inspect <project>       Detailed breakdown for one project.
    inspect <project> --scores   Same + mean/min/max per eval_function.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "evals"))
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "dbclients-project"
        / "src"
    ),
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def _counts(pool, project: str) -> dict:
    async with pool.acquire() as conn:
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1", project
        )
        gold = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1 AND reference IS NOT NULL",
            project,
        )
        runs = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project = $1", project
        )
        evals = await conn.fetchval(
            "SELECT COUNT(*) FROM eval_results WHERE project = $1", project
        )
    return {"docs": docs, "gold": gold, "runs": runs, "evals": evals}


async def _scores(pool, project: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT strategy, eval_function,
                   ROUND(AVG(score)::numeric, 3) AS mean,
                   ROUND(MIN(score)::numeric, 3) AS min,
                   ROUND(MAX(score)::numeric, 3) AS max
            FROM   eval_results
            WHERE  project = $1
            GROUP  BY strategy, eval_function
            ORDER  BY strategy, eval_function
            """,
            project,
        )
    return [dict(r) for r in rows]


async def cmd_status() -> None:
    from persist import _get_pool, ensure_tables
    pool = await _get_pool()
    await ensure_tables()

    from dataset import ConduitDatasetAsync
    projects = await ConduitDatasetAsync.list_projects(pool=pool)

    if not projects:
        console.print("[dim]No projects found.[/dim]")
        return

    table = Table(title="ConduitDataset Status", show_lines=True)
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Documents", justify="right")
    table.add_column("Gold Standards", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Evals", justify="right")

    for project in projects:
        c = await _counts(pool, project)
        pending = c["docs"] - c["gold"]
        gold_str = f"{c['gold']} / {c['docs']}" + (
            f"  [yellow]({pending} pending)[/yellow]" if pending else ""
        )
        table.add_row(project, str(c["docs"]), gold_str, str(c["runs"]), str(c["evals"]))

    console.print(table)


async def cmd_inspect(project: str, show_scores: bool) -> None:
    from persist import _get_pool, ensure_tables
    pool = await _get_pool()
    await ensure_tables()

    from dataset import ConduitDatasetAsync
    projects = await ConduitDatasetAsync.list_projects(pool=pool)
    if project not in projects:
        console.print(f"[red]Project '{project}' not found.[/red]")
        sys.exit(1)

    c = await _counts(pool, project)
    pending = c["docs"] - c["gold"]

    console.print(Panel(
        f"[bold cyan]{project}[/bold cyan]\n\n"
        f"  Documents : {c['docs']}  "
        f"({'[yellow]' + str(pending) + ' pending gold standard[/yellow]' if pending else '[green]all have gold standards[/green]'})\n"
        f"  Runs      : {c['runs']}\n"
        f"  Evals     : {c['evals']}",
        title="ConduitDataset Inspect",
        expand=False,
    ))

    if show_scores:
        score_rows = await _scores(pool, project)
        if not score_rows:
            console.print("[dim]No eval scores yet.[/dim]")
            return
        t = Table(title="Scores by Strategy × Eval Function", show_lines=True)
        t.add_column("Strategy", style="cyan")
        t.add_column("Eval Function", style="magenta")
        t.add_column("Mean", justify="right")
        t.add_column("Min", justify="right")
        t.add_column("Max", justify="right")
        for row in score_rows:
            t.add_row(
                row["strategy"], row["eval_function"],
                str(row["mean"]), str(row["min"]), str(row["max"]),
            )
        console.print(t)


def _get_pool_or_exit():
    """Resolve pool; exit 1 with clear message on connection failure."""
    from persist import _get_pool
    from dbclients.discovery.host import get_network_context

    loop = asyncio.new_event_loop()
    try:
        return loop, loop.run_until_complete(_get_pool())
    except Exception as e:
        ctx = get_network_context()
        console.print(
            f"[bold red]Cannot reach postgres at "
            f"{ctx.preferred_host}:5432 (database: evals)[/bold red]\n"
            f"[dim]{e}[/dim]"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="conduit-dataset",
        description="ConduitDataset inventory CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="List all projects with stage counts")

    ins = sub.add_parser("inspect", help="Detailed breakdown for one project")
    ins.add_argument("project", help="Project name")
    ins.add_argument(
        "--scores", action="store_true", help="Include score stats per eval function"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    loop = asyncio.new_event_loop()
    try:
        if args.command == "status":
            loop.run_until_complete(cmd_status())
        elif args.command == "inspect":
            loop.run_until_complete(cmd_inspect(args.project, args.scores))
    except SystemExit:
        raise
    except ConnectionError as e:
        from dbclients.discovery.host import get_network_context
        ctx = get_network_context()
        console.print(
            f"[bold red]Cannot reach postgres at "
            f"{ctx.preferred_host}:5432 (database: evals)[/bold red]\n"
            f"[dim]{e}[/dim]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(2)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.4: Run AC 16 test**
```bash
python -m pytest tests/test_dataset.py::test_cli_status_exits_1_when_db_unreachable -v
```
Expected: PASS

- [ ] **Step 10.5: Smoke-test the CLI manually**
```bash
cd /home/fishhouses/Brian_Code/conduit-project
python src/conduit/apps/scripts/datasets_cli.py status
python src/conduit/apps/scripts/datasets_cli.py inspect summarize
```
Expected: rich table output (or "No projects found" if DB is empty)

- [ ] **Step 10.6: Commit**
```bash
git add src/conduit/apps/scripts/datasets_cli.py evals/tests/test_dataset.py
git commit -m "feat: datasets_cli.py status and inspect commands — AC 16"
```

---

## Task 11: Register entrypoint in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 11.1: Add entrypoint**

In `pyproject.toml`, under `[project.scripts]`, add:
```toml
conduit-dataset = "conduit.apps.scripts.datasets_cli:main"
```

- [ ] **Step 11.2: Reinstall the package**
```bash
cd /home/fishhouses/Brian_Code/conduit-project
pip install -e . --quiet
```

- [ ] **Step 11.3: Verify the command is available**
```bash
conduit-dataset status
```
Expected: same output as Step 10.5

- [ ] **Step 11.4: Run full test suite to verify no regressions**
```bash
cd /home/fishhouses/Brian_Code/conduit-project/evals
python -m pytest tests/test_dataset.py -v
```
Expected: all 18 tests PASS

- [ ] **Step 11.5: Commit**
```bash
git add pyproject.toml
git commit -m "feat: register conduit-dataset CLI entrypoint in pyproject.toml"
```

---

## Completion Checklist

All 18 acceptance criteria covered:

| AC | Task | Test name |
|---|---|---|
| AC 1 | Task 3 | `test_documents_list_returns_run_inputs_with_reference` |
| AC 2 | Task 4 | `test_documents_swap_is_atomic` |
| AC 3 | Task 4 | `test_documents_delete_all_raises_if_runs_exist`, `test_documents_delete_all_force_clears_all` |
| AC 4 | Task 3 | `test_documents_save_does_not_overwrite_reference` |
| AC 5 | Task 5 | `test_save_gold_standards_raises_if_reference_exists`, `test_save_gold_standards_force_overwrites` |
| AC 6 | Task 5 | `test_save_gold_standards_raises_if_source_id_missing` |
| AC 7 | Task 5 | `test_save_gold_standards_skips_none_reference` |
| AC 8 | Task 7 | `test_runs_list_filters_by_config_dict` |
| AC 9 | Task 7 | `test_runs_list_unknown_config_returns_empty` |
| AC 10 | Task 7 | `test_runs_list_raises_on_conflicting_config_args` |
| AC 11 | Task 7 | `test_runs_list_configs_ordered_by_created_at` |
| AC 12 | Task 8 | `test_evals_save_per_result_transaction_and_batch_error` |
| AC 13 | Task 6 | `test_runs_save_deduplicates_configs` |
| AC 14 | Task 9 | `test_sync_raises_when_loop_is_running` |
| AC 15 | Task 9 | `test_sync_raises_after_close` |
| AC 16 | Task 10 | `test_cli_status_exits_1_when_db_unreachable` |
| AC 17 | Tasks 3, 6, 8 | `test_documents_save_is_idempotent`, `test_runs_save_is_idempotent`, `test_evals_save_is_idempotent` |
| AC 18 | Task 9 | `test_sync_context_manager_closes_on_exit`, `test_sync_context_manager_closes_on_exception` |
