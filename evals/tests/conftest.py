from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "dbclients-project" / "src"))


@pytest_asyncio.fixture
async def pool():
    from persist import _get_pool, ensure_tables
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
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM run_results WHERE project = $1", project)
        await conn.execute("DELETE FROM documents WHERE project = $1", project)


@pytest_asyncio.fixture
async def cd(project, cleanup, pool):
    from dataset import ConduitDatasetAsync
    return ConduitDatasetAsync(project, pool=pool)
