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

    async def ls_all(self) -> list[dict[str, object]]:
        """
        Return aggregate stats for every cache_name in the table.
        Each dict has exactly these keys:
            cache_name: str
            total_entries: int
            total_size_bytes: int
            oldest_record: str | None   # 'YYYY-MM-DD', None if no entries
            latest_record: str | None   # 'YYYY-MM-DD', None if no entries
        Ordered by cache_name ascending.
        Calls _ensure_ready() at entry.
        """
        pool = await self._ensure_ready()
        query = """
            SELECT
                cache_name,
                COUNT(*) AS total_entries,
                COALESCE(SUM(pg_column_size(payload)), 0) AS total_size_bytes,
                to_char(MIN(created_at), 'YYYY-MM-DD') AS oldest_record,
                to_char(MAX(updated_at), 'YYYY-MM-DD') AS latest_record
            FROM conduit_cache_entries
            GROUP BY cache_name
            ORDER BY cache_name ASC
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
        return [
            {
                "cache_name": row["cache_name"],
                "total_entries": row["total_entries"],
                "total_size_bytes": row["total_size_bytes"],
                "oldest_record": row["oldest_record"],
                "latest_record": row["latest_record"],
            }
            for row in rows
        ]

    async def delete_older_than(self, pg_interval: str) -> int:
        """
        Delete entries for this cache_name where created_at < now() - interval.
        `pg_interval` is a validated Postgres interval string (e.g. '7 days', '48 hours').
        Returns the number of rows deleted.
        Does NOT reset _hits, _misses, or _start_time.
        Calls _ensure_ready() at entry.
        """
        pool = await self._ensure_ready()
        query = """
            DELETE FROM conduit_cache_entries
            WHERE cache_name = $1
              AND created_at < now() - $2::interval
        """
        async with pool.acquire() as conn:
            result = await conn.execute(query, self.project_name, pg_interval)
        # asyncpg returns a status string like "DELETE 3"
        deleted = int(result.split()[-1])
        return deleted

    async def wipe_all(self) -> int:
        """
        Delete all rows in conduit_cache_entries with no cache_name filter.
        Returns the number of rows deleted.
        Calls _ensure_ready() at entry.
        """
        pool = await self._ensure_ready()
        query = "DELETE FROM conduit_cache_entries WHERE TRUE"
        async with pool.acquire() as conn:
            result = await conn.execute(query)
        deleted = int(result.split()[-1])
        return deleted

    async def inspect_latest(self) -> dict[str, object] | None:
        """
        Return the most recent entry for this cache_name, or None if empty.
        """
        pool = await self._ensure_ready()
        query = """
            SELECT cache_name, cache_key, payload, updated_at
            FROM conduit_cache_entries
            WHERE cache_name = $1
            ORDER BY updated_at DESC
            LIMIT 1
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, self.project_name)
        if row is None:
            return None
        return {
            "cache_name": row["cache_name"],
            "cache_key": row["cache_key"],
            "payload": row["payload"],
            "updated_at": row["updated_at"],
        }

    def _request_to_key(self, request: GenerationRequest) -> str:
        return request.generate_cache_key()
