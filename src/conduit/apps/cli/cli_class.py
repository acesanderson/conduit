"""
ConduitCLI is our conduit library as a CLI application.

Customize the query_function to specialize for various prompts / workflows while retaining archival and other functionalities.

To customize:
1. Define your own query function matching the QueryFunctionProtocol signature.
2. Pass your custom function to the ConduitCLI class upon instantiation. NOTE: all Conduit configs are namespaced in the query function inputs. (and any other handlers you define)
3. You can also pass other click options or commands to further customize the CLI behavior.

This allows you to tailor the behavior of ConduitCLI while leveraging its existing features.
"""

from conduit.config import settings
from conduit.apps.cli.query.query_function import (
    CLIQueryFunctionProtocol,
    default_query_function,
)
from conduit.apps.cli.commands.commands import CommandCollection
from conduit.apps.cli.utils.printer import Printer
import click
import sys
import logging

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_NAME = settings.default_project_name
DEFAULT_DESCRIPTION = "Conduit: The LLM CLI"
DEFAULT_QUERY_FUNCTION = default_query_function


class ConduitCLI:
    """
    Main class for the Conduit CLI application.
    Combines argument parsing, configuration loading, and command handling.
    Attributes:
        name (str): Name of the CLI application.
        description (str): Description of the CLI application.
        query_function (CLIQueryFunctionProtocol): Function to handle queries.
        verbosity (Verbosity): Verbosity level for LLM responses.
        cache (bool): Whether to use caching for LLM responses.
        persistent (bool): Whether to persist history and settings.
        system_message (str | None): System message for LLM context.
        preferred_model (str): Preferred LLM model to use.
    """

    def __init__(
        self,
        name: str = "conduit",
        description: str = DEFAULT_DESCRIPTION,
        query_function: CLIQueryFunctionProtocol = DEFAULT_QUERY_FUNCTION,
        version: str = settings.version,
    ):
        # Parameters
        self.name: str = name
        self.description: str = description
        self.query_function: CLIQueryFunctionProtocol = query_function
        # Components
        self.printer: Printer = Printer()
        self.cli: click.Group = self._build_cli()

    def _build_cli(self) -> click.Group:
        stdin: str = self._get_stdin()
        printer: Printer = self.printer

        @click.group(invoke_without_command=True)
        @click.option("--version", is_flag=True)
        @click.option("--raw", is_flag=True)
        @click.pass_context
        def cli(ctx, version, raw):
            ctx.ensure_object(dict)
            ctx.obj["stdin"] = stdin
            ctx.obj["printer"] = printer
            if raw:
                printer.set_raw(True)

            if version:
                click.echo(version)
                ctx.exit()

            if ctx.invoked_subcommand is None:
                click.echo(ctx.get_help())

        return cli

    def attach(self, command_collection: CommandCollection) -> None:
        """
        Attach a set of commands to the CLI.
        Args:
            command_set (click.Group): The set of commands to attach.
        """
        _ = command_collection.attach(self.cli)

    def run(self) -> None:
        self.cli()

    def _get_stdin(self) -> str:
        """
        Get implicit context from clipboard or other sources.
        """
        context = sys.stdin.read() if not sys.stdin.isatty() else ""
        return context
