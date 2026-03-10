# tests/cli/test_models_commands.py
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch, MagicMock

import click
import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli():
    """Minimal Click group wrapping models_command — avoids full ConduitCLI setup."""
    from conduit.apps.cli.commands.models_commands import models_command

    @click.group()
    def _cli():
        pass

    _cli.add_command(models_command)
    return _cli


def test_no_flags_calls_modelstore_display(runner, cli):
    """AC1: conduit models with no flags calls ModelStore.display()."""
    with patch("conduit.core.model.models.modelstore.ModelStore.display") as mock_display:
        result = runner.invoke(cli, ["models"])
        mock_display.assert_called_once()
        assert result.exit_code == 0


def test_no_module_level_modelstore_calls():
    """AC9: importing models_commands must not call any ModelStore methods."""
    sys.modules.pop("conduit.apps.cli.commands.models_commands", None)

    with patch("conduit.core.model.models.modelstore.ModelStore") as MockStore:
        importlib.import_module("conduit.apps.cli.commands.models_commands")
        MockStore.list_models.assert_not_called()
        MockStore.list_model_types.assert_not_called()
        MockStore.list_providers.assert_not_called()
        MockStore.display.assert_not_called()


def test_model_flag_calls_get_model(runner, cli):
    """AC4: -m with a known model calls ModelStore.get_model() and accesses .card."""
    from unittest.mock import PropertyMock

    mock_spec = MagicMock()
    card_mock = PropertyMock(return_value=None)
    type(mock_spec).card = card_mock

    with patch("conduit.core.model.models.modelstore.ModelStore.get_model", return_value=mock_spec):
        result = runner.invoke(cli, ["models", "-m", "claude-3-5-sonnet"])
        assert result.exit_code == 0
        card_mock.assert_called_once()


def test_model_flag_fuzzy_on_unknown_model(runner, cli):
    """AC4: -m with an unknown model prints fuzzy suggestions."""
    with patch("conduit.core.model.models.modelstore.ModelStore.get_model", side_effect=ValueError("not found")):
        with patch("conduit.core.model.models.modelstore.ModelStore.list_models", return_value=["claude-3-5-sonnet", "gpt-4o"]):
            with patch("rapidfuzz.process.extract", return_value=[("claude-3-5-sonnet", 90, 0)]):
                result = runner.invoke(cli, ["models", "-m", "claud"])
                assert result.exit_code == 0
                assert "Did you mean" in result.output


def test_type_flag_prints_filtered_models(runner, cli):
    """AC5: -t prints models filtered by type."""
    mock_spec = MagicMock()
    mock_spec.model = "claude-3-5-sonnet"

    with patch("conduit.core.model.models.modelstore.ModelStore.list_model_types", return_value=["chat", "embedding"]):
        with patch("conduit.core.model.models.modelstore.ModelStore.by_type", return_value=[mock_spec]) as mock_by_type:
            result = runner.invoke(cli, ["models", "-t", "chat"])
            mock_by_type.assert_called_once_with("chat")
            assert "claude-3-5-sonnet" in result.output
            assert result.exit_code == 0


def test_type_flag_rejects_invalid_type(runner, cli):
    """AC5: -t with an invalid type exits non-zero with an error message."""
    with patch("conduit.core.model.models.modelstore.ModelStore.list_model_types", return_value=["chat", "embedding"]):
        result = runner.invoke(cli, ["models", "-t", "nonexistent"])
        assert result.exit_code != 0
        assert "Must be one of" in result.output
