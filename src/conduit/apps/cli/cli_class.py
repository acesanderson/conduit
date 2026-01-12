"""
ConduitCLI is our conduit library as a CLI application.

Customize the query_function to specialize for various prompts / workflows while retaining archival and other functionalities.

To customize:
1. Define your own query function matching the QueryFunctionProtocol signature.
2. Pass your custom function to the ConduitCLI class upon instantiation. NOTE: all Conduit configs are namespaced in the query function inputs. (and any other handlers you define)
3. You can also pass other click options or commands to further customize the CLI behavior.

This allows you to tailor the behavior of ConduitCLI while leveraging its existing features.
"""

from __future__ import annotations
import asyncio
import sys
import click
import logging
from functools import cached_property
from typing import TYPE_CHECKING

from conduit.config import settings
from conduit.apps.cli.query.query_function import (
    CLIQueryFunctionProtocol,
    default_query_function,
)
from conduit.storage.repository.protocol import AsyncSessionRepository
from conduit.apps.cli.commands.commands import CommandCollection
from conduit.apps.cli.utils.printer import Printer

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation


logger = logging.getLogger(__name__)

# Defaults
DEFAULT_PROJECT_NAME = "conduit_cli"
DEFAULT_DESCRIPTION = "Conduit: The LLM CLI"
DEFAULT_QUERY_FUNCTION = default_query_function
PREFERRED_MODEL = settings.preferred_model
DEFAULT_SYSTEM_MESSAGE = settings.system_prompt


class ConduitCLI:
    """
    Main class for the Conduit CLI application.
    Combines argument parsing, configuration loading, and command handling.
    Attributes:
        project_name (str): Name of the CLI application.
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
        project_name: str = DEFAULT_PROJECT_NAME,
        description: str = DEFAULT_DESCRIPTION,
        query_function: CLIQueryFunctionProtocol = DEFAULT_QUERY_FUNCTION,
        model: str = PREFERRED_MODEL,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        version: str = settings.version,
    ):
        # Parameters
        self.project_name: str = project_name
        self.description: str = description
        self.query_function: CLIQueryFunctionProtocol = query_function
        self.version: str = version
        self.preferred_model: str = model
        self.system_message: str = system_message

        # Components
        self.printer: Printer = Printer()

        # --- Lifecycle Management ---
        # Create a persistent Event Loop for this CLI instance.
        # This bridges the synchronous CLI commands (Click) with the
        # asynchronous backend (Postgres/LLM).
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.cli: click.Group = self._build_cli()

    @cached_property
    def repository(self) -> AsyncSessionRepository:
        """
        Synchronously resolve the async repository.
        """
        from conduit.storage.repository.postgres_repository import (
            get_async_repository,
        )

        return get_async_repository(project_name=self.project_name)

    @cached_property
    def conversation(self) -> Conversation:
        """
        Load the last conversation or create a new one.
        """
        last_conversation = self.loop.run_until_complete(self.repository.last)

        if last_conversation is not None:
            return last_conversation
        else:
            from conduit.domain.conversation.conversation import Conversation

            return Conversation()

    def _build_cli(self) -> click.Group:
        stdin = self._get_stdin()
        printer = self.printer
        version_string: str = self.version

        # invoke_without_command=True allows us to handle --version
        # without triggering a "Missing command" error
        @click.group(invoke_without_command=True)
        @click.option("--version", "show_version", is_flag=True)
        @click.option("--raw", is_flag=True)
        @click.pass_context
        def cli(ctx, show_version, raw):
            ctx.ensure_object(dict)
            # Dependency Injection
            ctx.obj["project_name"] = self.project_name
            ctx.obj["stdin"] = stdin
            ctx.obj["printer"] = printer

            # Pass the loop in case commands need to run async tasks
            ctx.obj["loop"] = self.loop

            ctx.obj["repository"] = lambda: self.repository  # Lazy load
            ctx.obj["conversation"] = lambda: self.conversation  # Lazy load
            ctx.obj["query_function"] = self.query_function
            ctx.obj["preferred_model"] = self.preferred_model
            ctx.obj["system_message"] = self.system_message
            ctx.obj["verbosity"] = settings.default_verbosity

            # Global Handler: Raw Mode
            if raw:
                printer.set_raw(True)

            # Global Handler: Version
            if show_version:
                click.echo(version_string)
                ctx.exit()

            # Strict Logic: If no subcommand is provided, just show help.
            # We no longer guess that the user meant to query.
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
        try:
            self.cli()
        finally:
            # Clean up the event loop when the CLI exits
            if self.loop.is_running():
                self.loop.close()

    def _get_stdin(self) -> str:
        """
        Get implicit context from clipboard or other sources.
        """
        context = sys.stdin.read() if not sys.stdin.isatty() else ""
        return context
