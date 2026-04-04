# ModelSpec Postgres Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-machine TinyDB `modelspecs.json` with a shared Postgres table so `models -m <name>` works across all hosts without per-machine setup.

**Architecture:** A new `ModelSpecRepository` class in `src/conduit/storage/` wraps all Postgres CRUD for `ModelSpec` objects. `ModelStore` and `research_models` are updated to call this repository instead of `modelspecs_CRUD`. The sync bridge (`_run`) uses `asyncio.run()` with a db_manager singleton reset after each call — required because asyncpg pools are tied to the event loop they were created in. ModelSpec generation remains a manual CLI operation (`update`); `update_ollama` only reports coverage gaps.

**Tech Stack:** asyncpg (via existing `db_manager`/`dbclients`), pytest + AsyncMock, Click CliRunner

---

## Non-goals (do not implement)
- Auto-generating ModelSpecs during `update_ollama` — explicitly rejected
- Keeping `modelspecs.json` / TinyDB as a fallback — delete it entirely
- Migrating existing TinyDB data — start Postgres fresh; `update` regenerates
- HeadwaterServer endpoint changes — separate work item
- Any REST API, versioning, or audit log for ModelSpecs

---

## Acceptance Criteria (referenced in each task)

- **AC1**: After running `update` on machine A, `models -m <name>` succeeds on machine B without any command on B
- **AC2**: Running `update_ollama` does not add or modify any row in `model_specs`
- **AC3**: `models -m <name>` for a model not in Postgres prints fuzzy suggestions — no unhandled exception
- **AC4**: If Postgres is unreachable, `models -m <name>` exits with a clear error message, not a traceback
- **AC5**: `model_specs` table enforces a unique constraint on `model` at the DB level
- **AC6**: `update` is idempotent — running it twice produces identical Postgres state
- **AC7**: `update_ollama` output includes the count of ollama models with no ModelSpec

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `src/conduit/storage/modelspec_repository.py` | `ModelSpecRepository`, `ModelSpecRepositoryError` |
| **Create** | `tests/storage/test_modelspec_repository.py` | Unit tests for repository (async internals mocked) |
| **Modify** | `src/conduit/core/model/models/modelstore.py` | Replace all `modelspecs_CRUD` imports with repository calls |
| **Modify** | `src/conduit/core/model/models/research_models.py` | `create_modelspec()` and `create_from_scratch()` use repository |
| **Modify** | `src/conduit/apps/scripts/update_ollama_list.py` | Print count of ollama models missing ModelSpecs |
| **Modify** | `src/conduit/apps/scripts/export_heavy_models.py` | Replace `modelspecs_CRUD` import with repository |
| **Modify** | `src/conduit/apps/cli/commands/models_commands.py` | Catch `ModelSpecRepositoryError`, print clean error |
| **Modify** | `tests/cli/test_models_commands.py` | Add AC3 and AC4 tests |
| **Delete** | `src/conduit/core/model/models/modelspecs_CRUD.py` | Replaced entirely by repository |

---

## Task 1: Create `ModelSpecRepository` — schema and read operations

**Fulfills: AC5** (PRIMARY KEY enforces unique constraint at DB level)

**Files:**
- Create: `src/conduit/storage/modelspec_repository.py`
- Create: `tests/storage/test_modelspec_repository.py`

- [ ] **Step 1: Write failing tests for schema and read operations**

```python
# tests/storage/test_modelspec_repository.py
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.storage.modelspec_repository import ModelSpecRepository


def make_mock_pool_and_conn():
    """Return (pool, conn) where pool.acquire() is an async context manager."""
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool, conn


@pytest.mark.asyncio
async def test_ensure_schema_creates_table():
    """Schema init executes CREATE TABLE IF NOT EXISTS with PRIMARY KEY on model."""
    pool, conn = make_mock_pool_and_conn()
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        await repo._ensure_schema()
    sql = conn.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS model_specs" in sql
    assert "model TEXT PRIMARY KEY" in sql
    assert "created_at TIMESTAMPTZ" in sql
    assert "updated_at TIMESTAMPTZ" in sql


@pytest.mark.asyncio
async def test_get_all_returns_empty_list_when_no_rows():
    pool, conn = make_mock_pool_and_conn()
    conn.fetch.return_value = []
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        result = await repo._get_all()
    assert result == []


@pytest.mark.asyncio
async def test_get_all_deserializes_modelspecs():
    from conduit.core.model.models.modelspec import ModelSpec
    spec = ModelSpec(
        model="gpt-4o",
        description="Test model",
        provider="openai",
        temperature_range=[0.0, 2.0],
        context_window=128000,
        text_completion=True,
        image_analysis=True,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=False,
    )
    pool, conn = make_mock_pool_and_conn()
    conn.fetch.return_value = [{"data": json.dumps(spec.model_dump())}]
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        result = await repo._get_all()
    assert len(result) == 1
    assert result[0].model == "gpt-4o"
    assert result[0].provider == "openai"


@pytest.mark.asyncio
async def test_get_by_name_returns_none_when_not_found():
    pool, conn = make_mock_pool_and_conn()
    conn.fetchrow.return_value = None
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        result = await repo._get_by_name("nonexistent-model")
    assert result is None


@pytest.mark.asyncio
async def test_get_by_name_returns_modelspec_when_found():
    from conduit.core.model.models.modelspec import ModelSpec
    spec = ModelSpec(
        model="claude-sonnet-4-6",
        description="Test",
        provider="anthropic",
        temperature_range=[0.0, 1.0],
        context_window=200000,
        text_completion=True,
        image_analysis=True,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=False,
    )
    pool, conn = make_mock_pool_and_conn()
    conn.fetchrow.return_value = {"data": json.dumps(spec.model_dump())}
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        result = await repo._get_by_name("claude-sonnet-4-6")
    assert result is not None
    assert result.model == "claude-sonnet-4-6"
    conn.fetchrow.assert_called_once_with(
        "SELECT data FROM model_specs WHERE model = $1", "claude-sonnet-4-6"
    )


@pytest.mark.asyncio
async def test_get_all_names_returns_list_of_strings():
    pool, conn = make_mock_pool_and_conn()
    conn.fetch.return_value = [{"model": "gpt-4o"}, {"model": "claude-sonnet-4-6"}]
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        result = await repo._get_all_names()
    assert result == ["gpt-4o", "claude-sonnet-4-6"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/storage/test_modelspec_repository.py -v
```

Expected: `ImportError` — `modelspec_repository` does not exist yet.

- [ ] **Step 3: Create `ModelSpecRepository` with schema and read methods**

```python
# src/conduit/storage/modelspec_repository.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from conduit.storage.db_manager import db_manager

if TYPE_CHECKING:
    from conduit.core.model.models.modelspec import ModelSpec

logger = logging.getLogger(__name__)

_DB_NAME = "conduit"


class ModelSpecRepositoryError(Exception):
    """Raised when Postgres is unavailable or a repository operation fails."""


class ModelSpecRepository:
    """
    Postgres-backed store for ModelSpec objects.

    Public methods are synchronous. Internally they call asyncio.run() and
    reset the DatabaseManager singleton afterward — asyncpg pools are bound to
    the event loop they were created in, so each asyncio.run() call must start
    with a clean manager.
    """

    async def _ensure_schema(self) -> None:
        pool = await db_manager.get_pool(_DB_NAME)
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_specs (
                    model       TEXT PRIMARY KEY,
                    data        JSONB NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)

    async def _get_all(self) -> list[ModelSpec]:
        from conduit.core.model.models.modelspec import ModelSpec
        pool = await db_manager.get_pool(_DB_NAME)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT data FROM model_specs ORDER BY model")
        return [ModelSpec(**json.loads(r["data"])) for r in rows]

    async def _get_by_name(self, model: str) -> ModelSpec | None:
        from conduit.core.model.models.modelspec import ModelSpec
        pool = await db_manager.get_pool(_DB_NAME)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM model_specs WHERE model = $1", model
            )
        if row is None:
            return None
        return ModelSpec(**json.loads(row["data"]))

    async def _get_all_names(self) -> list[str]:
        pool = await db_manager.get_pool(_DB_NAME)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT model FROM model_specs ORDER BY model")
        return [r["model"] for r in rows]

    def _run(self, coro):
        """
        Run an async coroutine from sync code.

        Resets DatabaseManager singleton after each call so the next invocation
        gets a fresh asyncpg pool in a fresh event loop.
        """
        from conduit.storage.db_manager import DatabaseManager
        try:
            return asyncio.run(coro)
        except Exception as exc:
            raise ModelSpecRepositoryError(
                f"Postgres unavailable or operation failed: {exc}"
            ) from exc
        finally:
            DatabaseManager._instance = None

    def initialize(self) -> None:
        """Create the model_specs table if it does not exist."""
        self._run(self._ensure_schema())

    def get_all(self) -> list[ModelSpec]:
        """Return all ModelSpecs from Postgres."""
        return self._run(self._get_all())

    def get_by_name(self, model: str) -> ModelSpec | None:
        """Return the ModelSpec for a model name, or None if not found."""
        return self._run(self._get_by_name(model))

    def get_all_names(self) -> list[str]:
        """Return all model names stored in Postgres."""
        return self._run(self._get_all_names())
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/storage/test_modelspec_repository.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/storage/modelspec_repository.py tests/storage/test_modelspec_repository.py
git commit -m "feat: add ModelSpecRepository with Postgres schema and read methods"
```

---

## Task 2: Add `upsert()` and `delete()` to `ModelSpecRepository`

**Fulfills: AC5** (ON CONFLICT on PRIMARY KEY is the DB-level enforcement), **AC6** (upsert is idempotent by design)

**Files:**
- Modify: `src/conduit/storage/modelspec_repository.py`
- Modify: `tests/storage/test_modelspec_repository.py`

- [ ] **Step 1: Write failing tests for upsert and delete**

Add to `tests/storage/test_modelspec_repository.py`:

```python
@pytest.mark.asyncio
async def test_upsert_executes_insert_on_conflict_update():
    from conduit.core.model.models.modelspec import ModelSpec
    spec = ModelSpec(
        model="gpt-4o",
        description="Test",
        provider="openai",
        temperature_range=[0.0, 2.0],
        context_window=128000,
        text_completion=True,
        image_analysis=False,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=False,
    )
    pool, conn = make_mock_pool_and_conn()
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        await repo._upsert(spec)
    sql = conn.execute.call_args[0][0]
    assert "ON CONFLICT (model) DO UPDATE" in sql
    assert conn.execute.call_args[0][1] == "gpt-4o"


@pytest.mark.asyncio
async def test_upsert_is_idempotent_on_same_model():
    """Calling _upsert twice with the same model name should not raise."""
    from conduit.core.model.models.modelspec import ModelSpec
    spec = ModelSpec(
        model="gpt-4o",
        description="Test",
        provider="openai",
        temperature_range=[0.0, 2.0],
        context_window=128000,
        text_completion=True,
        image_analysis=False,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=False,
    )
    pool, conn = make_mock_pool_and_conn()
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        await repo._upsert(spec)
        await repo._upsert(spec)
    assert conn.execute.call_count == 2  # called twice, no exception


@pytest.mark.asyncio
async def test_delete_executes_delete_sql():
    pool, conn = make_mock_pool_and_conn()
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        await repo._delete("gpt-4o")
    sql = conn.execute.call_args[0][0]
    assert "DELETE FROM model_specs WHERE model = $1" in sql
    assert conn.execute.call_args[0][1] == "gpt-4o"


@pytest.mark.asyncio
async def test_delete_nonexistent_model_does_not_raise():
    """Deleting a model that is not in DB should silently succeed."""
    pool, conn = make_mock_pool_and_conn()
    conn.execute.return_value = None
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(return_value=pool)
        repo = ModelSpecRepository()
        await repo._delete("nonexistent")  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/storage/test_modelspec_repository.py::test_upsert_executes_insert_on_conflict_update tests/storage/test_modelspec_repository.py::test_delete_executes_delete_sql -v
```

Expected: `AttributeError` — `_upsert` and `_delete` not defined yet.

- [ ] **Step 3: Add `_upsert`, `_delete`, `upsert`, and `delete` to `ModelSpecRepository`**

Add to the class in `src/conduit/storage/modelspec_repository.py` (before `_run`):

```python
    async def _upsert(self, spec: ModelSpec) -> None:
        pool = await db_manager.get_pool(_DB_NAME)
        data_json = json.dumps(spec.model_dump())
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_specs (model, data, created_at, updated_at)
                VALUES ($1, $2::jsonb, NOW(), NOW())
                ON CONFLICT (model) DO UPDATE SET
                    data       = EXCLUDED.data,
                    updated_at = NOW()
                """,
                spec.model,
                data_json,
            )

    async def _delete(self, model: str) -> None:
        pool = await db_manager.get_pool(_DB_NAME)
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM model_specs WHERE model = $1", model
            )
```

Add public sync wrappers (after `get_all_names`):

```python
    def upsert(self, spec: ModelSpec) -> None:
        """Insert or update a ModelSpec in Postgres."""
        self._run(self._upsert(spec))

    def delete(self, model: str) -> None:
        """Remove a ModelSpec from Postgres. No-op if the model is not present."""
        self._run(self._delete(model))
```

- [ ] **Step 4: Run all repository tests**

```
pytest tests/storage/test_modelspec_repository.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/storage/modelspec_repository.py tests/storage/test_modelspec_repository.py
git commit -m "feat: add upsert and delete to ModelSpecRepository"
```

---

## Task 3: `ModelSpecRepositoryError` on Postgres unavailable

**Fulfills: AC4**

**Files:**
- Modify: `tests/storage/test_modelspec_repository.py`

- [ ] **Step 1: Write failing test for the error path**

Add to `tests/storage/test_modelspec_repository.py`:

```python
def test_run_wraps_connection_failure_in_repository_error():
    """If the DB is unreachable, _run() raises ModelSpecRepositoryError, not the raw exception."""
    from conduit.storage.modelspec_repository import ModelSpecRepositoryError
    with patch("conduit.storage.modelspec_repository.db_manager") as mock_dm:
        mock_dm.get_pool = AsyncMock(side_effect=OSError("Connection refused"))
        repo = ModelSpecRepository()
        with pytest.raises(ModelSpecRepositoryError, match="Postgres unavailable"):
            repo.get_all()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/storage/test_modelspec_repository.py::test_run_wraps_connection_failure_in_repository_error -v
```

Expected: FAIL — either raises `OSError` directly (not wrapped) or `ModelSpecRepositoryError` with wrong message.

- [ ] **Step 3: Verify `_run` already handles this**

The `_run` method written in Task 1 already wraps all exceptions. Check that the `match="Postgres unavailable"` string is present in the exception message in `_run`. If it is, the test passes already. If the wording differs, update `_run`'s message to include "Postgres unavailable":

```python
    def _run(self, coro):
        from conduit.storage.db_manager import DatabaseManager
        try:
            return asyncio.run(coro)
        except Exception as exc:
            raise ModelSpecRepositoryError(
                f"Postgres unavailable or operation failed: {exc}"
            ) from exc
        finally:
            DatabaseManager._instance = None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/storage/test_modelspec_repository.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/storage/test_modelspec_repository.py
git commit -m "test: verify ModelSpecRepositoryError wraps Postgres connection failures"
```

---

## Task 4: Wire `ModelStore.get_model()` to repository

**Fulfills: AC1, AC3, AC4**

**Files:**
- Modify: `src/conduit/core/model/models/modelstore.py`
- Modify: `tests/cli/test_models_commands.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/cli/test_models_commands.py`:

```python
def test_model_flag_shows_error_on_db_unavailable(runner, cli):
    """AC4: if Postgres is unreachable, models -m prints a clear error and exits non-zero."""
    from conduit.storage.modelspec_repository import ModelSpecRepositoryError
    with patch(
        "conduit.core.model.models.modelstore.ModelStore.get_model",
        side_effect=ModelSpecRepositoryError("Connection refused"),
    ):
        result = runner.invoke(cli, ["models", "-m", "gpt-4o"])
    assert result.exit_code != 0
    assert "Database unavailable" in result.output or "Postgres" in result.output


def test_model_flag_fuzzy_on_unknown_model_no_traceback(runner, cli):
    """AC3: models -m with an unknown model shows fuzzy suggestions and exits 0."""
    with patch(
        "conduit.core.model.models.modelstore.ModelStore.get_model",
        side_effect=ValueError("not found"),
    ):
        with patch(
            "conduit.core.model.models.modelstore.ModelStore.list_models",
            return_value=["gpt-4o", "claude-sonnet-4-6"],
        ):
            with patch(
                "rapidfuzz.process.extract",
                return_value=[("gpt-4o", 85, 0)],
            ):
                result = runner.invoke(cli, ["models", "-m", "gpt4"])
    assert result.exit_code == 0
    assert "Did you mean" in result.output
    assert "Traceback" not in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/cli/test_models_commands.py::test_model_flag_shows_error_on_db_unavailable tests/cli/test_models_commands.py::test_model_flag_fuzzy_on_unknown_model_no_traceback -v
```

Expected: `test_model_flag_shows_error_on_db_unavailable` FAILS — `ModelSpecRepositoryError` is not caught, leaks as a traceback.

- [ ] **Step 3: Update `ModelStore.get_model()` in `modelstore.py`**

Replace the existing `get_model` classmethod (lines ~334–344):

```python
    @classmethod
    def get_model(cls, model: str) -> ModelSpec:
        """
        Get the ModelSpec for a model name.
        Raises ValueError if not found, ModelSpecRepositoryError if Postgres is down.
        """
        from conduit.storage.modelspec_repository import ModelSpecRepository
        model = cls.validate_model(model)
        repo = ModelSpecRepository()
        result = repo.get_by_name(model)
        if result is None:
            raise ValueError(f"Model '{model}' not found in the database.")
        return result
```

- [ ] **Step 4: Catch `ModelSpecRepositoryError` in `models_command`**

In `src/conduit/apps/cli/commands/models_commands.py`, replace the `if model:` block (lines ~66–89):

```python
    if model:
        from conduit.core.model.models.modelstore import ModelStore
        from conduit.storage.modelspec_repository import ModelSpecRepositoryError
        from rich.console import Console

        console = Console()
        try:
            modelspec = ModelStore.get_model(model)
            modelspec.card
        except ModelSpecRepositoryError as exc:
            console.print(f"[red]Database unavailable: {exc}[/red]")
            raise SystemExit(1)
        except ValueError:
            from rapidfuzz import process
            from rapidfuzz import fuzz
            from collections import namedtuple

            Match = namedtuple("Match", ["title", "score", "rank"])
            models_list = ModelStore.list_models()
            results = process.extract(model, models_list, scorer=fuzz.WRatio, limit=3)
            matches = [
                Match(title=title, score=score, rank=rank + 1)
                for rank, (title, score, _) in enumerate(results)
            ]
            console.print(f"[red]Model '{model}' not found. Did you mean:[/red]")
            for match in matches:
                console.print(f"  {match.rank}. {match.title}")
        return
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/cli/test_models_commands.py -v
```

Expected: all tests PASS including the two new ones.

- [ ] **Step 6: Commit**

```bash
git add src/conduit/core/model/models/modelstore.py \
        src/conduit/apps/cli/commands/models_commands.py \
        tests/cli/test_models_commands.py
git commit -m "feat: wire ModelStore.get_model() to ModelSpecRepository; handle DB errors in models_command"
```

---

## Task 5: Wire `ModelStore` read methods to repository

**Fulfills: AC1** (data comes from shared Postgres, not per-machine file)

**Files:**
- Modify: `src/conduit/core/model/models/modelstore.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/cli/test_models_commands.py`:

```python
def test_provider_flag_ollama_calls_show_ollama_table(runner, cli):
    """AC1: models -p ollama path still works after ModelStore changes."""
    with patch("conduit.apps.cli.commands.models_commands._show_ollama_table") as mock_table:
        with patch(
            "conduit.core.model.models.modelstore.ModelStore.list_providers",
            return_value=["ollama", "openai"],
        ):
            result = runner.invoke(cli, ["models", "-p", "ollama"])
    mock_table.assert_called_once()
    assert result.exit_code == 0
```

And a unit test targeting `ModelStore` directly:

```python
# Add to tests/storage/test_modelspec_repository.py
def test_modelstore_get_all_models_uses_repository():
    """ModelStore.get_all_models() calls ModelSpecRepository.get_all(), not TinyDB."""
    from conduit.core.model.models.modelstore import ModelStore
    from conduit.core.model.models.modelspec import ModelSpec

    mock_spec = ModelSpec(
        model="gpt-4o",
        description="Test",
        provider="openai",
        temperature_range=[0.0, 2.0],
        context_window=128000,
        text_completion=True,
        image_analysis=False,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=False,
    )
    with patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.get_all",
        return_value=[mock_spec],
    ):
        result = ModelStore.get_all_models()
    assert len(result) == 1
    assert result[0].model == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/storage/test_modelspec_repository.py::test_modelstore_get_all_models_uses_repository -v
```

Expected: FAIL — `ModelStore.get_all_models()` still calls TinyDB.

- [ ] **Step 3: Replace remaining TinyDB calls in `modelstore.py`**

Replace `get_all_models()` (lines ~347–353):

```python
    @classmethod
    def get_all_models(cls) -> list[ModelSpec]:
        """Get all models as ModelSpec objects."""
        from conduit.storage.modelspec_repository import ModelSpecRepository
        return ModelSpecRepository().get_all()
```

Replace `by_provider()` (lines ~408–416):

```python
    @classmethod
    def by_provider(cls, provider: Provider) -> list[ModelSpec]:
        """Get a list of models for a specific provider."""
        from conduit.storage.modelspec_repository import ModelSpecRepository
        return [
            spec for spec in ModelSpecRepository().get_all()
            if spec.provider == provider
        ]
```

Replace `by_type()` — this method delegates to the capability-specific methods, which already call `get_all_models()`. No change needed beyond `get_all_models`.

Also remove the stale `TYPE_CHECKING` import of `get_all_modelspecs` from the top of `modelstore.py`. The block should be:

```python
if TYPE_CHECKING:
    from conduit.core.model.models.modelspec import ModelSpec
```

- [ ] **Step 4: Run full test suite**

```
pytest tests/storage/test_modelspec_repository.py tests/cli/test_models_commands.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/core/model/models/modelstore.py \
        tests/storage/test_modelspec_repository.py \
        tests/cli/test_models_commands.py
git commit -m "feat: wire ModelStore.get_all_models() and by_provider() to ModelSpecRepository"
```

---

## Task 6: Wire `ModelStore._is_consistent()` and `_update_models()` to repository

**Fulfills: AC6** (idempotency — upsert means running twice is safe)

**Files:**
- Modify: `src/conduit/core/model/models/modelstore.py`

- [ ] **Step 1: Write failing test for idempotency**

Add to `tests/storage/test_modelspec_repository.py`:

```python
def test_modelstore_update_is_idempotent():
    """
    AC6: Calling ModelStore._update_models() when all models are already in Postgres
    results in zero upserts and zero deletes.
    """
    from conduit.core.model.models.modelstore import ModelStore

    all_model_names = list(ModelStore.list_models())

    with patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.get_all_names",
        return_value=all_model_names,
    ) as mock_names, patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.delete"
    ) as mock_delete, patch(
        "conduit.core.model.models.research_models.create_modelspec"
    ) as mock_create:
        ModelStore._update_models()

    mock_delete.assert_not_called()
    mock_create.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/storage/test_modelspec_repository.py::test_modelstore_update_is_idempotent -v
```

Expected: FAIL — `_update_models()` still imports from `modelspecs_CRUD`.

- [ ] **Step 3: Replace TinyDB calls in `_is_consistent()` and `_update_models()`**

Replace `_update_models()` (lines ~262–300) in `modelstore.py`:

```python
    @classmethod
    def _update_models(cls):
        from conduit.core.model.models.research_models import create_modelspec
        from conduit.storage.modelspec_repository import ModelSpecRepository

        repo = ModelSpecRepository()
        modelspec_db_names = set(repo.get_all_names())
        models = cls.models()
        models_json_names = set(itertools.chain.from_iterable(models.values()))
        models_not_in_db = models_json_names - modelspec_db_names
        models_not_in_list = modelspec_db_names - models_json_names
        logger.info(f"Found {len(models_not_in_db)} models not in Postgres.")
        logger.info(f"Found {len(models_not_in_list)} stale models to remove from Postgres.")
        for model in models_not_in_list:
            repo.delete(model)
            logger.info(f"Deleted stale ModelSpec for {model}")
        for model in models_not_in_db:
            create_modelspec(model)
            logger.info(f"Created ModelSpec for {model}")
        if cls._is_consistent():
            logger.info("ModelSpecs are now consistent. Update complete.")
            return
        raise ValueError(
            "ModelSpecs not consistent after update — check Postgres and logs."
        )
```

Replace `_is_consistent()` (lines ~303–330):

```python
    @classmethod
    def _is_consistent(cls) -> bool:
        """Check if every model in models.json and the ollama cache has a ModelSpec in Postgres."""
        from conduit.storage.modelspec_repository import ModelSpecRepository

        repo = ModelSpecRepository()
        db_names = set(repo.get_all_names())
        models = cls.models()
        model_names = set(itertools.chain.from_iterable(models.values()))
        return model_names == db_names
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/storage/test_modelspec_repository.py tests/cli/test_models_commands.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/core/model/models/modelstore.py \
        tests/storage/test_modelspec_repository.py
git commit -m "feat: wire ModelStore._is_consistent() and _update_models() to ModelSpecRepository"
```

---

## Task 7: Wire `research_models.py` to repository

**Fulfills: AC6** (`upsert` makes `create_modelspec` idempotent)

**Files:**
- Modify: `src/conduit/core/model/models/research_models.py`

- [ ] **Step 1: Write failing test**

Add to `tests/storage/test_modelspec_repository.py`:

```python
def test_create_modelspec_uses_repo_upsert():
    """
    research_models.create_modelspec() must call ModelSpecRepository.upsert(),
    not TinyDB add_modelspec.
    """
    from conduit.core.model.models.modelspec import ModelSpec
    mock_spec = ModelSpec(
        model="qwen3:30b",
        description="Test",
        provider="ollama",
        temperature_range=[0.0, 1.0],
        context_window=32768,
        text_completion=True,
        image_analysis=False,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=False,
    )
    with patch(
        "conduit.core.model.models.research_models.get_capabilities_by_model",
        return_value=mock_spec,
    ), patch(
        "conduit.core.model.models.modelstore.ModelStore.identify_provider",
        return_value="ollama",
    ), patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.upsert"
    ) as mock_upsert:
        from conduit.core.model.models.research_models import create_modelspec
        create_modelspec("qwen3:30b")
    mock_upsert.assert_called_once()
    called_spec = mock_upsert.call_args[0][0]
    assert called_spec.model == "qwen3:30b"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/storage/test_modelspec_repository.py::test_create_modelspec_uses_repo_upsert -v
```

Expected: FAIL — `create_modelspec` still calls TinyDB.

- [ ] **Step 3: Rewrite `research_models.py` imports and affected functions**

Replace the top import block (lines 1–7) in `research_models.py`:

```python
from conduit.core.model.models.modelspec import ModelSpecList, ModelSpec
from conduit.core.model.model_sync import ModelSync as Model
from conduit.sync import Conduit, GenerationParams, ConduitOptions, Verbosity
from conduit.core.prompt.prompt import Prompt
from rich.console import Console
```

Replace `create_from_scratch()` (lines ~130–141):

```python
def create_from_scratch() -> None:
    """
    Rebuild the entire ModelSpec table in Postgres from scratch.
    Deletes all existing rows, then regenerates via Perplexity.
    """
    from conduit.storage.modelspec_repository import ModelSpecRepository

    repo = ModelSpecRepository()
    all_specs = get_all_capabilities()
    for name in repo.get_all_names():
        repo.delete(name)
    for spec in all_specs:
        repo.upsert(spec)
    print(f"Populated ModelSpecs table with {len(all_specs)} entries.")
    retrieved = repo.get_all()
    assert len(retrieved) == len(all_specs), (
        "Retrieved specs do not match created specs."
    )
```

Replace `create_modelspec()` (lines ~161–179):

```python
def create_modelspec(model: str) -> None:
    """
    Generate and persist a ModelSpec for a single model via Perplexity.
    Uses upsert — safe to call more than once for the same model.
    """
    from conduit.core.model.models.modelstore import ModelStore
    from conduit.storage.modelspec_repository import ModelSpecRepository

    provider = ModelStore.identify_provider(model)
    model_spec = get_capabilities_by_model(provider, model)
    if not isinstance(model_spec, ModelSpec):
        raise ValueError(
            f"Expected ModelSpec, got {type(model_spec)} for model {model}"
        )
    model_spec.model = model
    repo = ModelSpecRepository()
    repo.upsert(model_spec)
    print(f"Upserted ModelSpec for {model_spec.model} to Postgres.")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/storage/test_modelspec_repository.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/core/model/models/research_models.py \
        tests/storage/test_modelspec_repository.py
git commit -m "feat: wire research_models to ModelSpecRepository; upsert replaces TinyDB add/in_db"
```

---

## Task 8: Wire `export_heavy_models.py` to repository

**Files:**
- Modify: `src/conduit/apps/scripts/export_heavy_models.py`

- [ ] **Step 1: Write failing test**

Add to `tests/storage/test_modelspec_repository.py`:

```python
def test_export_heavy_models_uses_repository():
    """export_heavy_models.main() must not import from modelspecs_CRUD."""
    import sys
    sys.modules.pop("conduit.apps.scripts.export_heavy_models", None)

    from conduit.core.model.models.modelspec import ModelSpec

    heavy_spec = ModelSpec(
        model="qwq:latest",
        description="Heavy model",
        provider="ollama",
        temperature_range=[0.0, 1.0],
        context_window=32768,
        heavy=True,
        text_completion=True,
        image_analysis=False,
        image_gen=False,
        audio_analysis=False,
        audio_gen=False,
        video_analysis=False,
        video_gen=False,
        reasoning=True,
    )
    with patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.get_all",
        return_value=[heavy_spec],
    ):
        import io
        import sys as _sys
        captured = io.StringIO()
        _sys.stdout = captured
        try:
            from conduit.apps.scripts.export_heavy_models import main
            main()
        finally:
            _sys.stdout = _sys.__stdout__
    assert "qwq:latest" in captured.getvalue()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/storage/test_modelspec_repository.py::test_export_heavy_models_uses_repository -v
```

Expected: FAIL — `export_heavy_models` still imports from `modelspecs_CRUD`.

- [ ] **Step 3: Rewrite `export_heavy_models.py`**

```python
"""
Export the list of heavy Ollama models as YAML to stdout.

A model is "heavy" if heavy=True in its ModelSpec (>30B parameters or >24GB VRAM).

Usage:
    export_heavy_models            # writes YAML to stdout
    export_heavy_models > heavy.yaml

Output format:
    heavy_models:
      - qwq:latest
      - deepseek-r1:70b
"""

from __future__ import annotations

import sys

import yaml

from conduit.storage.modelspec_repository import ModelSpecRepository


def main() -> None:
    repo = ModelSpecRepository()
    all_specs = repo.get_all()
    heavy = sorted(
        spec.model for spec in all_specs if getattr(spec, "heavy", False)
    )
    yaml.dump({"heavy_models": heavy}, sys.stdout, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/storage/test_modelspec_repository.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/scripts/export_heavy_models.py \
        tests/storage/test_modelspec_repository.py
git commit -m "feat: wire export_heavy_models to ModelSpecRepository"
```

---

## Task 9: Update `update_ollama_list.py` — report missing ModelSpecs

**Fulfills: AC2** (no writes to Postgres), **AC7** (print missing-spec count)

**Files:**
- Modify: `src/conduit/apps/scripts/update_ollama_list.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/storage/test_modelspec_repository.py`:

```python
def test_update_ollama_prints_missing_spec_count(capsys):
    """
    AC7: update_ollama_list.main() prints how many ollama models lack a ModelSpec.
    AC2: it does NOT call ModelSpecRepository.upsert() or .delete().
    """
    from click.testing import CliRunner
    from conduit.apps.scripts.update_ollama_list import main

    runner = CliRunner()

    with patch(
        "conduit.core.clients.ollama.server_registry.fetch_server_models",
        new_callable=AsyncMock,
        return_value=["qwen3:30b", "gemma4:latest"],
    ), patch(
        "conduit.apps.scripts.update_ollama_list.write_server_to_cache"
    ), patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.get_all_names",
        return_value=["qwen3:30b"],  # gemma4:latest has no spec
    ) as mock_names, patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.upsert"
    ) as mock_upsert, patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.delete"
    ) as mock_delete, patch(
        "conduit.core.model.models.modelstore.ModelStore.models",
        return_value={"ollama": ["qwen3:30b", "gemma4:latest"]},
    ):
        result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert "1" in result.output  # 1 model missing spec
    assert "no ModelSpec" in result.output or "missing" in result.output.lower()
    mock_upsert.assert_not_called()
    mock_delete.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/storage/test_modelspec_repository.py::test_update_ollama_prints_missing_spec_count -v
```

Expected: FAIL — `update_ollama_list.main()` does not print the missing count.

- [ ] **Step 3: Add missing-spec reporting to `update_ollama_list.py`**

In `src/conduit/apps/scripts/update_ollama_list.py`, after the final `console.print` of the model list, add:

```python
    # Report coverage gap — do NOT write to Postgres here.
    try:
        from conduit.storage.modelspec_repository import ModelSpecRepository
        from conduit.storage.modelspec_repository import ModelSpecRepositoryError

        repo = ModelSpecRepository()
        spec_names = set(repo.get_all_names())
        ollama_models = set(all_models)
        missing = ollama_models - spec_names
        if missing:
            console.print(
                f"\n[yellow]{len(missing)} ollama model(s) have no ModelSpec[/yellow]"
                " — run [cyan]update[/cyan] to generate"
            )
        else:
            console.print("\n[green]All ollama models have ModelSpecs.[/green]")
    except ModelSpecRepositoryError:
        console.print("[dim]Could not reach Postgres to check ModelSpec coverage.[/dim]")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/storage/test_modelspec_repository.py::test_update_ollama_prints_missing_spec_count -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/scripts/update_ollama_list.py \
        tests/storage/test_modelspec_repository.py
git commit -m "feat: update_ollama reports ollama models with no ModelSpec (AC2, AC7)"
```

---

## Task 10: Delete `modelspecs_CRUD.py` and clean up remaining imports

**Files:**
- Delete: `src/conduit/core/model/models/modelspecs_CRUD.py`
- Verify: no remaining imports in `modelstore.py` or elsewhere

- [ ] **Step 1: Verify no remaining references**

```bash
grep -r "modelspecs_CRUD\|tinydb\|TinyDB" src/conduit/ tests/
```

Expected output: **no matches**. If any appear, fix them before proceeding.

If `modelstore.py` still has any local `from conduit.core.model.models.modelspecs_CRUD import ...` inside methods, replace each with the corresponding `ModelSpecRepository` call per Tasks 4–6.

- [ ] **Step 2: Delete the file**

```bash
git rm src/conduit/core/model/models/modelspecs_CRUD.py
```

- [ ] **Step 3: Run full test suite to confirm nothing is broken**

```
pytest tests/storage/ tests/cli/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: delete modelspecs_CRUD.py — fully replaced by ModelSpecRepository"
```

---

## Task 11: Run full test suite and smoke test

- [ ] **Step 1: Run all tests**

```
pytest -v
```

Expected: all tests PASS. No `ImportError` or `TinyDB` references.

- [ ] **Step 2: Smoke test the `models` CLI locally**

```bash
# Schema must auto-create on first use:
python -c "from conduit.storage.modelspec_repository import ModelSpecRepository; ModelSpecRepository().initialize()"

# List all models (should show 0 modelspecs until update runs):
models

# Run update to populate Postgres from models.json + ollama cache:
update

# Verify a known cloud model now works:
models -m gpt-4o

# Verify unknown model gives fuzzy suggestions, not traceback:
models -m gpt4ohh

# Verify ollama listing still works (reads from cache, not Postgres):
models -p ollama
```

- [ ] **Step 3: Commit if any smoke-test fixes were needed**

```bash
git add -p
git commit -m "fix: smoke-test corrections for ModelSpec Postgres migration"
```

---

## Task 12: Deployment to deepwater, bywater, backwater

**Fulfills: AC1** (shared Postgres — verified cross-machine)

The `model_specs` table lives in the `conduit` Postgres database on Caruana (bywater), which all hosts already connect to. Deployment means: update the installed package on each host, then run `update` once to populate the shared table.

- [ ] **Step 1: Deploy to deepwater (AlphaBlue)**

```bash
ssh alphablue "cd ~/Brian_Code/conduit-project && git pull && uv pip install -e . --quiet"
```

Verify:

```bash
ssh alphablue "python -c 'from conduit.storage.modelspec_repository import ModelSpecRepository; print(\"OK\")'"
```

Expected: `OK`

- [ ] **Step 2: Deploy to bywater (Caruana)**

```bash
ssh caruana "cd ~/Brian_Code/conduit-project && git pull && uv pip install -e . --quiet"
```

Verify:

```bash
ssh caruana "python -c 'from conduit.storage.modelspec_repository import ModelSpecRepository; print(\"OK\")'"
```

Expected: `OK`

- [ ] **Step 3: Deploy to backwater (Cheet)**

```bash
ssh cheet "cd ~/Brian_Code/conduit-project && git pull && uv pip install -e . --quiet"
```

Verify:

```bash
ssh cheet "python -c 'from conduit.storage.modelspec_repository import ModelSpecRepository; print(\"OK\")'"
```

Expected: `OK`

- [ ] **Step 4: Populate the shared Postgres table (run once, from any host)**

```bash
update
```

This calls `ModelStore.update()` → `_update_models()` → fires Perplexity for each model not yet in Postgres. This is the expensive one-time step. Watch the output — it should log one line per model created.

Expected output includes lines like:
```
Upserted ModelSpec for gpt-4o to Postgres.
Upserted ModelSpec for claude-sonnet-4-6 to Postgres.
...
Model specifications are now consistent with models.json. Update complete.
```

- [ ] **Step 5: Verify AC1 cross-machine**

From a machine that did NOT run `update`, confirm that `models -m gpt-4o` works:

```bash
# Example: run from caruana if update was run on alphablue
ssh caruana "models -m gpt-4o"
```

Expected: model card printed, exit 0. No `ValueError`, no traceback.

- [ ] **Step 6: Verify update_ollama now reports ModelSpec coverage**

```bash
update_ollama
```

Expected output includes something like:
```
Total cached Ollama models: N
  ...list of models...

0 ollama models have no ModelSpec — run update to generate
```

(Or `N ollama models have no ModelSpec` if ollama models are not yet specced — that is expected; run `update` again to generate them.)

- [ ] **Step 7: Final commit / tag**

```bash
git tag modelspec-postgres-v1
git push && git push --tags
```

---

## Self-Review

**Spec coverage check:**

| AC | Covered by |
|----|-----------|
| AC1 (cross-machine) | Tasks 4–7 (Postgres read/write), Task 12 (deployment verification) |
| AC2 (update_ollama no writes) | Task 9 test asserts `upsert` and `delete` not called |
| AC3 (fuzzy suggestions, no traceback) | Task 4 test and implementation |
| AC4 (DB unavailable → clean error) | Tasks 3 + 4 |
| AC5 (unique constraint at DB level) | Task 1 schema (PRIMARY KEY), Task 2 upsert uses ON CONFLICT |
| AC6 (idempotency) | Task 6 test + upsert-based implementation |
| AC7 (update_ollama prints count) | Task 9 |

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency check:**
- `ModelSpecRepository.get_by_name()` → `ModelSpec | None` — used correctly in `ModelStore.get_model()` (None check before return)
- `ModelSpecRepository.get_all_names()` → `list[str]` — used correctly in `_is_consistent()` and `_update_models()`
- `ModelSpecRepositoryError` imported from `conduit.storage.modelspec_repository` consistently in all callers

**Missing items resolved:**
- `export_heavy_models.py` was an additional caller not in the original notes — covered in Task 8
- `modelspecs_CRUD.create_modelspec` broken stub (lone `t` on line 39) — eliminated by Task 10
