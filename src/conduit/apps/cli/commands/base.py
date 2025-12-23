from __future__ import annotations
import click
from conduit.apps.cli.commands.commands import CommandCollection
from conduit.apps.cli.handlers.base_handlers import BaseHandlers
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    from conduit.storage.repository.protocol import ConversationRepository
    from conduit.apps.cli.utils.printer import Printer
    from uuid import UUID

handlers = BaseHandlers()


class BaseCommands(CommandCollection):
    """
    Attachable command collection for Conduit CLI.
    Inject onto any Click group via `attach(group)`.
    """

    def __init__(self):
        self._commands: list[click.Command] = []
        self._register_commands()

    @override
    def _register_commands(self):
        """Define all the base commands as bound methods."""

        @click.command()
        @click.option("-m", "--model", type=str, help="Specify the model to use.")
        @click.option("-L", "--local", is_flag=True, help="Use local HeadwaterServer.")
        @click.option("-r", "--raw", is_flag=True, help="Print raw output.")
        @click.option("-t", "--temperature", type=float, help="Temperature (0.0-1.0).")
        @click.option(
            "-c", "--chat", is_flag=True, help="Enable chat mode with history."
        )
        @click.option("-a", "--append", type=str, help="Append to query after stdin.")
        @click.option(
            "-p", "--prepend", type=str, help="Prepend to query before stdin."
        )
        @click.argument("query_input", nargs=-1)
        @click.pass_context
        def query(
            ctx: click.Context,
            model: str,
            local: bool,
            raw: bool,
            temperature: float,
            chat: bool,
            append: str,
            prepend: str,
            query_input: str,
        ):
            """Execute a query (default command)."""
            handlers.handle_query(
                ctx,
                model,
                local,
                raw,
                temperature,
                chat,
                append,
                prepend,
                query_input,
            )

        @click.command()
        def history(ctx: click.Context):
            """View message history."""
            repository: ConversationRepository = ctx.obj["repository"]
            conversation_id: str | UUID = ctx.obj["conversation_id"]
            printer: Printer = ctx.obj["printer"]

            handlers.handle_history(repository, conversation_id, printer)

        @click.command()
        def wipe(ctx: click.Context):
            """Wipe message history."""
            repository: ConversationRepository = ctx.obj["repository"]
            conversation_id: str | UUID = ctx.obj["conversation_id"]
            printer: Printer = ctx.obj["printer"]

            handlers.handle_wipe(printer, repository, conversation_id)

        @click.command()
        def ping(ctx: click.Context):
            """Ping the Headwater server."""
            printer: Printer = ctx.obj["printer"]

            handlers.handle_ping(printer)

        @click.command()
        def status(ctx: click.Context):
            """Get Headwater server status."""
            printer: Printer = ctx.obj["printer"]

            handlers.handle_status(printer)

        @click.command()
        def shell(ctx: click.Context):
            """Enter interactive shell mode."""
            raise NotImplementedError

        @click.command()
        def last(ctx: click.Context):
            """Get the last message."""
            conversation: Conversation = ctx.obj["conversation"]
            printer: Printer = ctx.obj["printer"]

            handlers.handle_last(printer, conversation)

        @click.command()
        @click.argument("index", type=int)
        def get(ctx: click.Context, index: int):
            """Get a specific message from history."""
            conversation: Conversation = ctx.obj["conversation"]
            printer: Printer = ctx.obj["printer"]

            handlers.handle_get(index, conversation, printer)

        @click.command()
        def config(ctx: click.Context):
            """View current configuration."""
            printer: Printer = ctx.obj["printer"]
            preferred_model: str = ctx.obj["preferred_model"]
            system_message: str = ctx.obj["system_message"]
            chat: bool = ctx.obj["chat"]
            verbosity: Verbosity = ctx.obj["verbosity"]

            handlers.handle_config(
                printer,
                preferred_model,
                system_message,
                chat,
                verbosity,
            )

        self._commands = [
            query,
            history,
            wipe,
            ping,
            status,
            shell,
            last,
            get,
            config,
        ]

    @override
    def attach(self, group: click.Group) -> click.Group:
        """Attach all commands to a Click group."""
        for cmd in self._commands:
            group.add_command(cmd)
        return group
