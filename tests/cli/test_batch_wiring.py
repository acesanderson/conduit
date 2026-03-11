from __future__ import annotations
from click.testing import CliRunner
from conduit.apps.cli.cli_class import ConduitCLI
from conduit.apps.cli.commands.base_commands import BaseCommands
from conduit.apps.cli.commands.cache_commands import cache
from conduit.apps.cli.commands.models_commands import models_command
from conduit.apps.cli.commands.batch_commands import batch_command


def _build_cli():
    cli_app = ConduitCLI()
    cli_app.attach(BaseCommands())
    cli_app.cli.add_command(cache)
    cli_app.cli.add_command(models_command)
    cli_app.cli.add_command(batch_command)
    return cli_app.cli


def test_batch_subcommand_appears_in_help():
    cli = _build_cli()
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "batch" in result.output
