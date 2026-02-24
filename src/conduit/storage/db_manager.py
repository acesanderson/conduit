from __future__ import annotations
import asyncio
import logging
from typing import ClassVar

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Singleton Manager for Postgres Connection Pools.
    Handles lazy-initialization of both the Pool and the Lock within the active loop.
    """

    _instance: ClassVar[DatabaseManager | None] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize instance-level state
            cls._instance._pool = None
            cls._instance._lock = None
        return cls._instance

    def _get_lock(self) -> asyncio.Lock:
        """Ensure the lock is created within the current running loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_pool(self, db_name: str = "conduit"):
        """Get or create the shared connection pool."""
        # Use the lazy lock to prevent the 'thundering herd'
        async with self._get_lock():
            # Check if pool exists and is functional
            if self._pool is not None:
                return self._pool

            logger.info(f"Initializing shared Postgres connection pool [db={db_name}]")
            from dbclients.clients.postgres import get_postgres_client

            self._pool = await get_postgres_client(
                client_type="async", dbname=db_name
            )
            return self._pool

    async def shutdown(self):
        """Graceful asynchronous shutdown and state reset."""
        if self._lock is None:
            return

        async with self._lock:
            if self._pool:
                logger.info("Closing shared Postgres connection pool...")
                await self._pool.close()
                self._pool = None

            # Reset lock to None so a new one is created if the app restarts
            self._lock = None


# 2. Re-adding the global convenience instance for module-level imports
db_manager = DatabaseManager()
