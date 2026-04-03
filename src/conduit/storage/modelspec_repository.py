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

    def _run(self, coro):
        """
        Run an async coroutine from sync code.

        Resets the db_manager pool and lock after each asyncio.run() call so the
        next invocation gets a fresh asyncpg pool in a fresh event loop. asyncpg
        pools are bound to the event loop they were created in — without this
        reset, the next asyncio.run() call would try to reuse a pool from the
        closed event loop.
        """
        try:
            return asyncio.run(coro)
        except Exception as exc:
            raise ModelSpecRepositoryError(
                f"Postgres unavailable or operation failed: {exc}"
            ) from exc
        finally:
            db_manager._pool = None
            db_manager._lock = None

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

    def upsert(self, spec: ModelSpec) -> None:
        """Insert or update a ModelSpec in Postgres."""
        self._run(self._upsert(spec))

    def delete(self, model: str) -> None:
        """Remove a ModelSpec from Postgres. No-op if the model is not present."""
        self._run(self._delete(model))
