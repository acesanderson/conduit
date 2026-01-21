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

# CHANGE: Import the concrete class directly for proper typing and access
from conduit.storage.repository.postgres_repository import (
    AsyncPostgresSessionRepository,
)
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
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.cli: click.Group = self._build_cli()

    @cached_property
    def repository(self) -> AsyncPostgresSessionRepository:
        """
        Instantiate the repository.
        Note: The connection is NOT opened here. It is opened in run().
        """
        return AsyncPostgresSessionRepository(project_name=self.project_name)

    @cached_property
    def conversation(self) -> Conversation:
        """
        Load the last conversation or create a new one.
        """
        # This relies on self.repository being "open" (via run's context management)
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

            # Pass the loop so commands can run async tasks
            ctx.obj["loop"] = self.loop

            ctx.obj["repository"] = lambda: self.repository  # Lazy load
            ctx.obj["conversation"] = lambda: self.conversation  # Lazy load
            ctx.obj["query_function"] = self.query_function
            ctx.obj["preferred_model"] = self.preferred_model
            ctx.obj["system_message"] = self.system_message
            ctx.obj["verbosity"] = settings.default_verbosity

            if raw:
                printer.set_raw(True)

            if show_version:
                click.echo(version_string)
                ctx.exit()

            if ctx.invoked_subcommand is None:
                click.echo(ctx.get_help())

        return cli

    def attach(self, command_collection: CommandCollection) -> None:
        """
        Attach a set of commands to the CLI.
        """
        _ = command_collection.attach(self.cli)

    def run(self) -> None:
        """
        Execute the CLI.
        Manages the lifecycle of the Async Repository connection pool using the synchronous event loop.
        """
        # 1. Initialize Repository Connection
        # We manually trigger the context manager's entry point
        self.loop.run_until_complete(self.repository.__aenter__())

        try:
            self.cli()
        except Exception as e:
            # Catch-all to ensure we don't crash without closing,
            # though Click usually handles its own exceptions.
            logger.error(f"CLI Error: {e}")
            raise
        finally:
            # 2. Cleanup Repository Connection
            # This runs even if sys.exit() is called by a Click command
            if not self.loop.is_closed():
                # Manually trigger the context manager's exit point
                self.loop.run_until_complete(
                    self.repository.__aexit__(None, None, None)
                )
                self.loop.close()

    def _get_stdin(self) -> str:
        """
        Get implicit context from clipboard or other sources.
        """
        context = sys.stdin.read() if not sys.stdin.isatty() else ""
        return context
