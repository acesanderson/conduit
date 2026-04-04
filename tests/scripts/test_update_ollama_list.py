from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_update_ollama_prints_missing_spec_count(runner):
    """
    AC7: After cache update, output includes count of ollama models lacking a ModelSpec.
    AC2: ModelSpecRepository.upsert() and .delete() are NOT called.
    """
    with patch(
        "conduit.core.clients.ollama.server_registry.fetch_server_models",
        new_callable=AsyncMock,
        return_value=["qwen3:30b", "gemma4:latest"],
    ), patch(
        "conduit.apps.scripts.update_ollama_list.write_server_to_cache"
    ), patch(
        "conduit.core.model.models.modelstore.ModelStore.models",
        return_value={"ollama": ["qwen3:30b", "gemma4:latest"]},
    ), patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.get_all_names",
        return_value=["qwen3:30b"],  # gemma4:latest has no spec
    ), patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.upsert"
    ) as mock_upsert, patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.delete"
    ) as mock_delete:
        from conduit.apps.scripts.update_ollama_list import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0
    # AC7: count of missing models is printed
    assert "1" in result.output
    assert any(
        phrase in result.output
        for phrase in ["no ModelSpec", "missing", "have no ModelSpec"]
    )
    # AC2: no writes to Postgres
    mock_upsert.assert_not_called()
    mock_delete.assert_not_called()


def test_update_ollama_handles_postgres_unavailable(runner):
    """If Postgres is unreachable, the script prints a warning but still exits 0."""
    from conduit.storage.modelspec_repository import ModelSpecRepositoryError

    with patch(
        "conduit.core.clients.ollama.server_registry.fetch_server_models",
        new_callable=AsyncMock,
        return_value=["qwen3:30b"],
    ), patch(
        "conduit.apps.scripts.update_ollama_list.write_server_to_cache"
    ), patch(
        "conduit.core.model.models.modelstore.ModelStore.models",
        return_value={"ollama": ["qwen3:30b"]},
    ), patch(
        "conduit.storage.modelspec_repository.ModelSpecRepository.get_all_names",
        side_effect=ModelSpecRepositoryError("DB down"),
    ):
        from conduit.apps.scripts.update_ollama_list import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert any(
        phrase in result.output
        for phrase in ["Postgres", "reach", "Could not", "unavailable"]
    )
