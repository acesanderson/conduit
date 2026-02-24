from __future__ import annotations
import time
import json
import logging
from typing import TYPE_CHECKING
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
from conduit.storage.db_manager import db_manager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from asyncpg import Pool


class AsyncPostgresCache:
    """
    Async Postgres-backed cache using the shared DatabaseManager pool.
    """

    def __init__(self, project_name: str, db_name: str = "conduit"):
        self.project_name = project_name
        self.db_name = db_name
        self._hits = 0
        self._misses = 0
        self._start_time = time.time()
        self._schema_initialized = False

    async def _ensure_ready(self) -> Pool:
        if not self._schema_initialized:
            await self._initialize_schema(await db_manager.get_pool(self.db_name))
            self._schema_initialized = True
        return await db_manager.get_pool(self.db_name)

    async def _initialize_schema(self, pool: Pool) -> None:
        """Ensure schema exists. Safe to call repeatedly."""
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conduit_cache_entries (
                    cache_name  text      NOT NULL,
                    cache_key   text      NOT NULL,
                    payload     jsonb     NOT NULL,
                    created_at  timestamptz NOT NULL DEFAULT now(),
                    updated_at  timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (cache_name, cache_key)
                );
            """)

    async def get(self, request: GenerationRequest) -> GenerationResponse | None:
        pool = await self._ensure_ready()
        key = self._request_to_key(request)

        query = """
            SELECT payload
            FROM conduit_cache_entries
            WHERE cache_name = $1 AND cache_key = $2
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, self.project_name, key)

        if row is None:
            self._misses += 1
            return None

        self._hits += 1
        payload = json.loads(row["payload"])
        return GenerationResponse.model_validate(payload)

    async def get_all(self) -> list[GenerationResponse]:
        pool = await self._ensure_ready()

        query = """
            SELECT payload
            FROM conduit_cache_entries
            WHERE cache_name = $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, self.project_name)

        responses = []
        for row in rows:
            payload = json.loads(row["payload"])
            responses.append(GenerationResponse.model_validate(payload))

        return responses

    async def set(
        self, request: GenerationRequest, response: GenerationResponse
    ) -> None:
        pool = await self._ensure_ready()
        key = self._request_to_key(request)
        payload = response.model_dump_json()

        query = """
            INSERT INTO conduit_cache_entries (cache_name, cache_key, payload)
            VALUES ($1, $2, $3)
            ON CONFLICT (cache_name, cache_key)
            DO UPDATE SET
                payload = EXCLUDED.payload,
                updated_at = now()
        """

        async with pool.acquire() as conn:
            await conn.execute(query, self.project_name, key, payload)

    async def wipe(self) -> None:
        pool = await self._ensure_ready()
        query = "DELETE FROM conduit_cache_entries WHERE cache_name = $1"
        async with pool.acquire() as conn:
            await conn.execute(query, self.project_name)

        self._hits = 0
        self._misses = 0
        self._start_time = time.time()

    async def cache_stats(self) -> dict[str, object]:
        pool = await self._ensure_ready()
        async with pool.acquire() as conn:
            row_count = await conn.fetchval(
                "SELECT COUNT(*) FROM conduit_cache_entries WHERE cache_name = $1",
                self.project_name,
            )
            size_bytes = await conn.fetchval(
                "SELECT COALESCE(SUM(pg_column_size(payload)), 0) FROM conduit_cache_entries WHERE cache_name = $1",
                self.project_name,
            )
            bounds = await conn.fetchrow(
                """
                SELECT
                    to_char(MIN(created_at), 'YYYY-MM-DD') as oldest,
                    to_char(MAX(updated_at), 'YYYY-MM-DD') as latest
                FROM conduit_cache_entries
                WHERE cache_name = $1
            """,
                self.project_name,
            )

        uptime_seconds = time.time() - self._start_time

        return {
            "cache_name": self.project_name,
            "total_entries": row_count,
            "total_size_bytes": size_bytes,
            "uptime_seconds": uptime_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "oldest_record": bounds["oldest"] if bounds else None,
            "latest_record": bounds["latest"] if bounds else None,
        }

    def _request_to_key(self, request: GenerationRequest) -> str:
        return request.generate_cache_key()
