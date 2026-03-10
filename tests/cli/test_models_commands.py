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


def test_no_module_level_modelstore_calls():
    """AC9: importing models_commands must not call any ModelStore methods."""
    sys.modules.pop("conduit.apps.cli.commands.models_commands", None)

    with patch("conduit.core.model.models.modelstore.ModelStore") as MockStore:
        importlib.import_module("conduit.apps.cli.commands.models_commands")
        MockStore.list_models.assert_not_called()
        MockStore.list_model_types.assert_not_called()
        MockStore.list_providers.assert_not_called()
        MockStore.display.assert_not_called()
