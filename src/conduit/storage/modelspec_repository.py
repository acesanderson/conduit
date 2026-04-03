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
