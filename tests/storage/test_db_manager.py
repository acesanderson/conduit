from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.storage.db_manager import DatabaseManager, db_manager


# Fixtures
# ---

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton state between tests so each test gets a clean instance."""
    yield
    DatabaseManager._instance = None


def make_mock_pool() -> MagicMock:
    pool = MagicMock()
    pool.close = AsyncMock()
    return pool


# Singleton behavior
# ---

def test_singleton_returns_same_instance():
    a = DatabaseManager()
    b = DatabaseManager()
    assert a is b


def test_initial_state_is_clean():
    manager = DatabaseManager()
    assert manager._pool is None
    assert manager._lock is None


def test_global_instance_is_database_manager():
    assert isinstance(db_manager, DatabaseManager)


# get_pool behavior
# ---

@pytest.mark.asyncio
async def test_get_pool_returns_pool():
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool):
        manager = DatabaseManager()
        pool = await manager.get_pool()
        assert pool is mock_pool


@pytest.mark.asyncio
async def test_get_pool_creates_lock():
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool):
        manager = DatabaseManager()
        assert manager._lock is None
        await manager.get_pool()
        assert isinstance(manager._lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_get_pool_reuses_existing_pool():
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool) as mock_client:
        manager = DatabaseManager()
        pool1 = await manager.get_pool()
        pool2 = await manager.get_pool()
        assert pool1 is pool2
        mock_client.assert_called_once()


@pytest.mark.asyncio
async def test_get_pool_thundering_herd():
    """20 concurrent get_pool() calls must only initialize the pool once."""
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool) as mock_client:
        manager = DatabaseManager()
        pools = await asyncio.gather(*[manager.get_pool() for _ in range(20)])
        mock_client.assert_called_once()
        assert all(p is mock_pool for p in pools)


@pytest.mark.asyncio
async def test_get_pool_passes_db_name():
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool) as mock_client:
        manager = DatabaseManager()
        await manager.get_pool(db_name="my_db")
        _, kwargs = mock_client.call_args
        assert kwargs["dbname"] == "my_db"


# shutdown behavior
# ---

@pytest.mark.asyncio
async def test_shutdown_noop_when_never_initialized():
    """shutdown() before any get_pool() call should return without error."""
    manager = DatabaseManager()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_calls_pool_close():
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool):
        manager = DatabaseManager()
        await manager.get_pool()
        await manager.shutdown()
        mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_resets_pool_and_lock():
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool):
        manager = DatabaseManager()
        await manager.get_pool()
        await manager.shutdown()
        assert manager._pool is None
        assert manager._lock is None


@pytest.mark.asyncio
async def test_shutdown_idempotent():
    """Calling shutdown() twice should not raise."""
    mock_pool = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock, return_value=mock_pool):
        manager = DatabaseManager()
        await manager.get_pool()
        await manager.shutdown()
        await manager.shutdown()


# Post-shutdown re-initialization
# ---

@pytest.mark.asyncio
async def test_get_pool_after_shutdown_creates_new_pool():
    """After shutdown, get_pool() initializes a fresh pool."""
    mock_pool_1 = make_mock_pool()
    mock_pool_2 = make_mock_pool()
    with patch("dbclients.clients.postgres.get_postgres_client", new_callable=AsyncMock) as mock_client:
        mock_client.side_effect = [mock_pool_1, mock_pool_2]
        manager = DatabaseManager()

        pool1 = await manager.get_pool()
        assert pool1 is mock_pool_1

        await manager.shutdown()

        pool2 = await manager.get_pool()
        assert pool2 is mock_pool_2
        assert mock_client.call_count == 2
