from __future__ import annotations
import time
import json
import logging
import asyncio
from typing import TYPE_CHECKING, ClassVar
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from asyncpg import Pool


class AsyncPostgresCache:
    """
    Async Postgres-backed cache with SHARED connection pool.

    Key Design:
    - Class-level pool registry keyed by (db_name, event_loop_id)
    - All cache instances for the same DB share ONE pool per event loop
    - Automatic pool cleanup when event loops close
    """

    # Class-level pool registry: {(db_name, loop_id): Pool}
    _shared_pools: ClassVar[dict[tuple[str, int], Pool]] = {}
    _pool_locks: ClassVar[dict[tuple[str, int], asyncio.Lock]] = {}

    def __init__(self, project_name: str, db_name: str = "conduit"):
        self.project_name = project_name
        self.db_name = db_name
        self._hits = 0
        self._misses = 0
        self._start_time = time.time()

        # Instance no longer owns a pool
        self._pool_key: tuple[str, int] | None = None

    async def _get_pool(self) -> Pool:
        """
        Get or create a SHARED connection pool for this database and event loop.
        Thread-safe via async locks.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "AsyncPostgresCache must be used within an async context"
            )

        loop_id = id(current_loop)
        pool_key = (self.db_name, loop_id)

        # Fast path: pool exists and loop is still valid
        if pool_key in self._shared_pools:
            pool = self._shared_pools[pool_key]
            if not pool._closed:  # asyncpg pool check
                self._pool_key = pool_key
                return pool
            else:
                # Clean up dead pool
                del self._shared_pools[pool_key]
                if pool_key in self._pool_locks:
                    del self._pool_locks[pool_key]

        # Slow path: need to create pool (with locking to prevent duplicates)
        if pool_key not in self._pool_locks:
            self._pool_locks[pool_key] = asyncio.Lock()

        async with self._pool_locks[pool_key]:
            # Double-check: another task might have created it while we waited
            if pool_key in self._shared_pools:
                pool = self._shared_pools[pool_key]
                if not pool._closed:
                    self._pool_key = pool_key
                    return pool

            # Actually create the pool
            logger.info(
                f"Creating SHARED asyncpg pool for cache db='{self.db_name}' (loop {loop_id})"
            )
            from dbclients.clients.postgres import get_postgres_client

            pool = await get_postgres_client(client_type="async", dbname=self.db_name)
            self._shared_pools[pool_key] = pool
            self._pool_key = pool_key

            # Initialize schema on first connection
            await self._initialize_schema(pool)

            return pool

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
        pool = await self._get_pool()
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
        pool = await self._get_pool()

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
        pool = await self._get_pool()
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
        pool = await self._get_pool()
        query = "DELETE FROM conduit_cache_entries WHERE cache_name = $1"
        async with pool.acquire() as conn:
            await conn.execute(query, self.project_name)

        self._hits = 0
        self._misses = 0
        self._start_time = time.time()

    async def cache_stats(self) -> dict[str, object]:
        pool = await self._get_pool()
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

    # Context manager support for async usage
    async def aclose(self) -> None:
        """
        Close the connection pool associated with this cache instance's loop.
        Removes it from the class-level shared registry.
        """
        if self._pool_key and self._pool_key in self._shared_pools:
            pool = self._shared_pools[self._pool_key]
            try:
                await pool.close()
                logger.info(f"Closed shared pool for {self._pool_key}")
            except Exception as e:
                logger.warning(f"Error closing cache pool: {e}")
            finally:
                # Remove from registry so we don't try to reuse a closed pool
                self._shared_pools.pop(self._pool_key, None)
                self._pool_locks.pop(self._pool_key, None)
                self._pool_key = None

    async def __aenter__(self) -> AsyncPostgresCache:
        """Initialize the pool on entry."""
        await self._get_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the pool is closed on exit."""
        await self.aclose()
