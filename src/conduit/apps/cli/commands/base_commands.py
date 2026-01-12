"""
The Command layer defines Click commands and routes to handlers.
Click context is passed to handlers via ctx. Click context should not leak outside of this layer.
If context needs to be edited (like with conversation state management), it should be done here.
"""

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

        @click.command(no_args_is_help=True)
        @click.option(
            "-m", "--model", type=str, help="Specify the model to use.", default=None
        )
        @click.option("-L", "--local", is_flag=True, help="Use local HeadwaterServer.")
        @click.option("-r", "--raw", is_flag=True, help="Print raw output.")
        @click.option("-t", "--temperature", type=float, help="Temperature (0.0-1.0).")
        @click.option(
            "-c", "--chat", is_flag=True, help="Enable chat mode with history."
        )
        @click.option(
            "-a",
            "--append",
            type=str,
            help="Append to query after stdin.",
            default=None,
        )
        @click.argument("query_input", nargs=-1)
        @click.pass_context
        def query(
            ctx: click.Context,
            model: str | None,
            local: bool,
            raw: bool,
            temperature: float | None,
            chat: bool,
            append: str | None,
            query_input: tuple[str, ...],
        ):
            """
            Execute a query against the LLM.

            Input can be passed as arguments or piped via stdin.

            Examples:
                conduit query "Why is the sky blue?"
                conduit query Explain quantum computing --model gpt-4
                cat file.py | conduit query "Refactor this code"
            """
            # 1. Unpack Dependencies from Context
            # The command layer is responsible for knowing WHERE things live (ctx.obj)
            printer = ctx.obj["printer"]
            query_function = ctx.obj["query_function"]
            verbosity = ctx.obj["verbosity"]

            # 2. Extract Config / State
            stdin = ctx.obj.get("stdin")
            system_message = ctx.obj.get("system_message")
            project_name = ctx.obj.get("project_name")
            preferred_model = ctx.obj.get("preferred_model")

            # 3. Resolve Logic (Boundary Responsibility)
            # Determine the final model string here, so the handler is deterministic
            resolved_model = model or preferred_model or "gpt-4o"

            # Smudge query arguments
            query_input_str = " ".join(query_input).strip()

            # 4. Delegate to Handler
            handlers.handle_query(
                query_input=query_input_str,
                model=resolved_model,
                local=local,
                raw=raw,
                temperature=temperature,
                chat=chat,
                append=append,
                # Injected Dependencies
                printer=printer,
                query_function=query_function,
                stdin=stdin,
                system_message=system_message,
                verbosity=verbosity,
                project_name=project_name,
            )

        @click.command()
        @click.pass_context
        def history(ctx: click.Context):
            """View message history."""
            repository: ConversationRepository = ctx.obj["repository"]()  # Lazy load
            conversation: Conversation = ctx.obj["conversation"]()  # Lazy load

            # Correctly extract session ID
            session_id = None
            if conversation.session:
                session_id = conversation.session.session_id

            printer: Printer = ctx.obj["printer"]
            loop = ctx.obj["loop"]

            if not session_id:
                printer.print_pretty(
                    "[yellow]No active session to view history for.[/yellow]"
                )
                return

            handlers.handle_history(repository, session_id, printer, loop)

        @click.command()
        @click.pass_context
        def wipe(ctx: click.Context):
            """Wipe message history."""
            repository: ConversationRepository = ctx.obj["repository"]()  # Lazy load
            conversation = ctx.obj["conversation"]()  # Lazy load
            printer: Printer = ctx.obj["printer"]
            loop = ctx.obj["loop"]

            # Correctly extract session ID
            session_id = None
            if conversation.session:
                session_id = conversation.session.session_id

            if not session_id:
                printer.print_pretty(
                    "[yellow]No active session found to wipe.[/yellow]"
                )
                # We might want to wipe ALL sessions if none is active, but that's a different command
                # For now, let's treat this as a no-op
                return

            handlers.handle_wipe(printer, repository, session_id, loop)

            # Reset conversation in context
            from conduit.domain.conversation.conversation import Conversation

            ctx.obj["conversation"] = Conversation()
            # Note: We can't save here easily without running async, and we just wiped the ID.
            # The next query will start fresh.

        @click.command()
        @click.pass_context
        def ping(ctx: click.Context):
            """Ping the Headwater server."""
            printer: Printer = ctx.obj["printer"]

            handlers.handle_ping(printer)

        @click.command()
        @click.pass_context
        def status(ctx: click.Context):
            """Get Headwater server status."""
            printer: Printer = ctx.obj["printer"]

            handlers.handle_status(printer)

        @click.command()
        @click.pass_context
        def shell(ctx: click.Context):
            """Enter interactive shell mode."""
            raise NotImplementedError

        @click.command()
        @click.pass_context
        def last(ctx: click.Context):
            """Get the last message."""
            conversation: Conversation = ctx.obj["conversation"]()
            printer: Printer = ctx.obj["printer"]

            handlers.handle_last(printer, conversation)

        @click.command()
        @click.pass_context
        @click.argument("index", type=int)
        def get(ctx: click.Context, index: int):
            """Get a specific message from history."""
            conversation: Conversation = ctx.obj["conversation"]
            printer: Printer = ctx.obj["printer"]

            handlers.handle_get(index, conversation, printer)

        @click.command()
        @click.pass_context
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
