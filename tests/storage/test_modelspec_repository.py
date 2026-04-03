from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.model.models.modelspec import ModelSpec


# --- Test data ---

def make_spec() -> ModelSpec:
    return ModelSpec(
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


def make_mock_pool(conn: AsyncMock) -> MagicMock:
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


# --- Tests ---

@pytest.mark.asyncio
async def test_ensure_schema_creates_table():
    conn = AsyncMock()
    pool = make_mock_pool(conn)

    with patch("conduit.storage.modelspec_repository.db_manager") as mock_db:
        mock_db.get_pool = AsyncMock(return_value=pool)

        from conduit.storage.modelspec_repository import ModelSpecRepository
        repo = ModelSpecRepository()
        await repo._ensure_schema()

    conn.execute.assert_called_once()
    sql = conn.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS model_specs" in sql
    assert "model" in sql and "TEXT PRIMARY KEY" in sql
    assert "created_at" in sql and "TIMESTAMPTZ" in sql
    assert "updated_at" in sql and "TIMESTAMPTZ" in sql


@pytest.mark.asyncio
async def test_get_all_returns_empty_list_when_no_rows():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    pool = make_mock_pool(conn)

    with patch("conduit.storage.modelspec_repository.db_manager") as mock_db:
        mock_db.get_pool = AsyncMock(return_value=pool)

        from conduit.storage.modelspec_repository import ModelSpecRepository
        repo = ModelSpecRepository()
        result = await repo._get_all()

    assert result == []


@pytest.mark.asyncio
async def test_get_all_deserializes_modelspecs():
    spec = make_spec()
    row = {"data": spec.model_dump_json()}

    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[row])
    pool = make_mock_pool(conn)

    with patch("conduit.storage.modelspec_repository.db_manager") as mock_db:
        mock_db.get_pool = AsyncMock(return_value=pool)

        from conduit.storage.modelspec_repository import ModelSpecRepository
        repo = ModelSpecRepository()
        result = await repo._get_all()

    assert len(result) == 1
    assert result[0].model == "gpt-4o"
    assert result[0].provider == "openai"


@pytest.mark.asyncio
async def test_get_by_name_returns_none_when_not_found():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    pool = make_mock_pool(conn)

    with patch("conduit.storage.modelspec_repository.db_manager") as mock_db:
        mock_db.get_pool = AsyncMock(return_value=pool)

        from conduit.storage.modelspec_repository import ModelSpecRepository
        repo = ModelSpecRepository()
        result = await repo._get_by_name("nonexistent-model")

    assert result is None


@pytest.mark.asyncio
async def test_get_by_name_returns_modelspec_when_found():
    spec = make_spec()
    row = {"data": spec.model_dump_json()}

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)
    pool = make_mock_pool(conn)

    with patch("conduit.storage.modelspec_repository.db_manager") as mock_db:
        mock_db.get_pool = AsyncMock(return_value=pool)

        from conduit.storage.modelspec_repository import ModelSpecRepository
        repo = ModelSpecRepository()
        result = await repo._get_by_name("gpt-4o")

    assert result is not None
    assert result.model == "gpt-4o"
    # Verify the SQL parameter was passed correctly
    conn.fetchrow.assert_called_once()
    call_args = conn.fetchrow.call_args[0]
    assert call_args[1] == "gpt-4o"


@pytest.mark.asyncio
async def test_get_all_names_returns_list_of_strings():
    rows = [{"model": "gpt-4o"}, {"model": "claude-3-opus"}]

    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)
    pool = make_mock_pool(conn)

    with patch("conduit.storage.modelspec_repository.db_manager") as mock_db:
        mock_db.get_pool = AsyncMock(return_value=pool)

        from conduit.storage.modelspec_repository import ModelSpecRepository
        repo = ModelSpecRepository()
        result = await repo._get_all_names()

    assert isinstance(result, list)
    assert result == ["gpt-4o", "claude-3-opus"]
