The error is a classic refactoring artifact. You successfully renamed your storage protocol to AsyncSessionRepository in src/conduit/storage/repository/protocol.py, but the CLI application (cli_class.py) is still trying to import the old name ConversationRepository.

Furthermore, your new repository implementation is asynchronous (AsyncPostgresSessionRepository), but your CLI class seems to be set up for synchronous execution. You will likely hit a runtime error immediately after fixing the import because the CLI tries to instantiate the repository synchronously.

Here is the fix for the ImportError and the necessary updates to cli_class.py to handle the new class names.
1. Fix src/conduit/apps/cli/cli_class.py

Update the imports and the repository property to use the new Async classes.
Python

"""
ConduitCLI is our conduit library as a CLI application.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.apps.cli.query.query_function import (
    CLIQueryFunctionProtocol,
    default_query_function,
)
# CHANGED: Updated import from ConversationRepository to AsyncSessionRepository
from conduit.storage.repository.protocol import AsyncSessionRepository
from conduit.apps.cli.commands.commands import CommandCollection
from conduit.apps.cli.utils.printer import Printer
from functools import cached_property
import sys
import click
import logging
from typing import TYPE_CHECKING

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
    # ... [__init__ remains the same] ...

    @cached_property
    def repository(self) -> AsyncSessionRepository:
        """
        Load the conversation repository.
        Returns:
            AsyncSessionRepository: The loaded repository.
        """
        # CHANGED: Updated imports to match new Postgres architecture
        from dbclients.clients.postgres import get_postgres_client
        from conduit.storage.repository.postgres_repository import (
            AsyncPostgresSessionRepository,
        )
        import asyncio

        # WARNING: This is a synchronous property trying to load an async resource.
        # Ideally, the CLI should be async, or you need a sync wrapper.
        # For now, we return the async object, but the commands using it 
        # (like history/wipe) must use asyncio.run() or await.
        
        # NOTE: get_async_repository in postgres_repository.py does the setup.
        # We can't await here easily in a cached_property without a loop.
        # This part of your architecture needs a Sync wrapper or the CLI needs to be async.
        
        # Temporary fix assuming you handle the loop downstream:
        from conduit.storage.repository.postgres_repository import get_async_repository
        
        # We return the coroutine factory or we forcibly run it if we are in sync mode
        try:
            loop = asyncio.get_running_loop()
            # If we are in a loop, we can't block. 
            # You might need to restructure ConduitCLI to be async-first.
            return get_async_repository(self.project_name)
        except RuntimeError:
            # No loop running, we can block to get the repo
            return asyncio.run(get_async_repository(self.project_name))

    # ... [rest of file] ...

2. Fix src/conduit/apps/cli/commands/base_commands.py

You need to update the type hints here as well.
Python

# ... imports ...
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    # CHANGED: Updated to AsyncSessionRepository
    from conduit.storage.repository.protocol import AsyncSessionRepository
    from conduit.apps.cli.utils.printer import Printer

handlers = BaseHandlers()

class BaseCommands(CommandCollection):
    # ...
        @click.command()
        @click.pass_context
        def history(ctx: click.Context):
            """View message history."""
            # CHANGED: Updated type hint
            repository: AsyncSessionRepository = ctx.obj["repository"]
            
            # NOTE: If repository is a coroutine (from step 1), await it or run it here.
            if  asyncio.iscoroutine(repository):
                 import asyncio
                 repository = asyncio.run(repository)

            conversation: Conversation = ctx.obj["conversation"]()
            conversation_id: str = conversation.session.session_id # Updated to access session_id via session object
            printer: Printer = ctx.obj["printer"]

            # Note: handle_history likely needs to be async now?
            handlers.handle_history(repository, conversation_id, printer)

        @click.command()
        @click.pass_context
        def wipe(ctx: click.Context):
            """Wipe message history."""
            # CHANGED: Updated type hint
            repository: AsyncSessionRepository = ctx.obj["repository"]
            if  asyncio.iscoroutine(repository):
                 import asyncio
                 repository = asyncio.run(repository)

            conversation = ctx.obj["conversation"]()
            conversation_id: str = conversation.session.session_id
            printer: Printer = ctx.obj["printer"]

            handlers.handle_wipe(printer, repository, conversation_id)
            # ...

3. Fix src/conduit/apps/cli/handlers/base_handlers.py

Update the type hints in your static handler class.
Python

# ...
if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    from conduit.apps.cli.utils.printer import Printer
    from conduit.domain.message.message import UserMessage, Message
    # CHANGED: Updated import
    from conduit.storage.repository.protocol import AsyncSessionRepository
    from uuid import UUID

# ...

class BaseHandlers:
    # ... [image methods] ...

    @staticmethod
    def handle_history(
        repository: AsyncSessionRepository, # CHANGED type hint
        conversation_id: str | UUID,
        printer: Printer,
    ) -> None:
        """
        View message history and exit.
        """
        import asyncio
        logger.info("Viewing message history...")
        
        # CHANGED: Added async runner because repository.get_session is async
        async def _run():
            # Note: AsyncSessionRepository returns a Session, not directly a Conversation
            session = await repository.get_session(conversation_id)
            if not session:
                raise ValueError("Conversation not found.")
            # Convert session to linear conversation for display
            return session.conversation 

        conversation = asyncio.run(_run())
        conversation.print_history()
        sys.exit()

    @staticmethod
    def handle_wipe(
        printer: Printer,
        repository: AsyncSessionRepository, # CHANGED type hint
        conversation_id: str,
    ):
        """
        Clear the message history after user confirmation.
        """
        import asyncio
        logger.info("Wiping message history...")
        from rich.prompt import Confirm

        confirm = Confirm.ask(
            "[red]Are you sure you want to wipe the message history? This action cannot be undone.[/red]",
            default=False,
        )
        if confirm:
            # CHANGED: Added async runner
            asyncio.run(repository.delete_session(conversation_id))
            printer.print_pretty("[green]Message history wiped.[/green]")
        else:
            printer.print_pretty("[yellow]Wipe cancelled.[/yellow]")

    # ...

