"""
This is a static class containing CLI command handlers for the Conduit application.
- all methods must have the signature
- if the config file specifies a type for a command or flag, it must be one of: str, int, float, bool, and the method must handle that type accordingly
- the method names must match the handler names in the config file exactly, a la "handle_history", "handle_wipe", etc.
"""

from __future__ import annotations
from conduit.config import settings
from typing import TYPE_CHECKING
import logging
import sys

if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    from conduit.apps.cli.utils.printer import Printer
    from conduit.domain.message.message import UserMessage, Message
    from conduit.storage.repository.protocol import ConversationRepository
    from uuid import UUID

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VERBOSITY = settings.default_verbosity


class BaseHandlers:
    @staticmethod
    def grab_image_from_clipboard(printer: Printer) -> tuple[str, str] | None:
        """
        Attempt to grab image from clipboard; return tuple of mime_type and base64.
        """
        logger.info("Attempting to grab image from clipboard...")
        import os

        if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
            printer.print_pretty("Image paste not available over SSH.", style="red")
            return

        import warnings
        from PIL import ImageGrab
        import base64
        import io

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress PIL warnings
            image = ImageGrab.grabclipboard()

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            # Save for next query
            printer.print_pretty("Image captured!", style="green")
            # Build our ImageMessage
            image_content = img_base64
            mime_type = "image/png"
            return mime_type, image_content
        else:
            printer.print_pretty("No image detected.", style="red")

            sys.exit()

    @staticmethod
    def create_image_message(
        combined_query: str, mime_type: str, image_content: str
    ) -> UserMessage | None:
        logger.info("Creating image message...")
        if not image_content or not mime_type:
            return

        from conduit.domain.message.message import (
            UserMessage,
            ImageContent,
            TextContent,
        )

        text_content_obj = TextContent(text=combined_query)

        image_content_obj = ImageContent(
            url=f"data:{mime_type};base64,{image_content}",
            detail="auto",
        )

        content = [text_content_obj, image_content_obj]

        imagemessage = UserMessage(
            content=content,
        )
        return imagemessage

    # Handlers -- should match config file exactly (per validate_handlers)
    @staticmethod
    def handle_history(
        repository: ConversationRepository,
        conversation_id: str | UUID,
        printer: Printer,
    ) -> None:
        """
        View message history and exit.
        """
        logger.info("Viewing message history...")
        conversation = repository.load_by_conversation_id(conversation_id)
        messages = conversation.messages if conversation else []
        output = "# Message History\n\n"
        if not messages:
            printer.print_pretty("[red]No messages in history.[/red]")
            sys.exit()
        for i, message in enumerate(messages):
            output += f"## Message {i + 1}\n\n"
            output += str(message) + "\n\n"
        printer.print_markdown(output)
        sys.exit()

    @staticmethod
    def handle_wipe(
        printer: Printer,
        repository: ConversationRepository,
        conversation_id: str | UUID,
    ):
        """
        Clear the message history after user confirmation.
        """
        logger.info("Wiping message history...")
        from rich.prompt import Confirm

        confirm = Confirm.ask(
            "[red]Are you sure you want to wipe the message history? This action cannot be undone.[/red]",
            default=False,
        )
        if confirm:
            repository.remove_by_conversation_id(conversation_id)
            printer.print_pretty("[green]Message history wiped.[/green]")
        else:
            printer.print_pretty("[yellow]Wipe cancelled.[/yellow]")

    @staticmethod
    def handle_shell():
        pass

    @staticmethod
    def handle_ping(printer: Printer):
        from headwater_client.client.headwater_client import HeadwaterClient

        hc = HeadwaterClient()
        response = hc.ping()
        if response == True:
            response = "Pong!"
            printer.print_pretty(f"[green]{response}[/green]")
        else:
            response = "No response."
            printer.print_pretty(f"[red]{response}[/red]")

    @staticmethod
    def handle_status(printer: Printer):
        from headwater_client.client.headwater_client import HeadwaterClient

        hc = HeadwaterClient()
        status = hc.get_status()
        printer.print_pretty(status)

    @staticmethod
    def handle_last(
        printer: Printer,
        conversation: Conversation,
    ):
        """
        Print the last message in the message store and exit.
        """
        logger.info("Viewing last message...")
        import sys

        # Get last message
        last_message: Message = conversation.last()
        # If no messages, inform user
        if not last_message:
            printer.print_pretty("[red]No messages in history.[/red]")
            sys.exit()
        # Print last message
        printer.print_markdown(str(last_message))
        sys.exit()

    @staticmethod
    def handle_get(index: int, conversation: Conversation, printer: Printer):
        """
        Print a specific message from history and exit.
        """
        logger.info(f"Viewing message at index {index}...")
        import sys

        messages = conversation.messages
        # Validate index
        if index < 0 or index >= len(messages):
            printer.print_pretty("[red]Invalid message index.[/red]")
            sys.exit()
        # Get message
        message: Message = messages[index]
        # Print message
        printer.print_markdown(str(message))
        sys.exit()

    @staticmethod
    def handle_config(
        printer: Printer,
        preferred_model: str,
        system_message: str,
        chat: bool,
        verbosity: Verbosity,
    ):
        """
        Print the current configuration and exit.
        """
        logger.info("Viewing configuration...")
        config_md = f"""
# Current Configuration
| Setting | Value |
|---------|-------|
| Preferred Model | {preferred_model} |
| System Message | {system_message[:50]}... |
| Message History | {"Enabled" if chat else "Disabled"} |
| Verbosity | {verbosity} |
"""
        printer.print_markdown(config_md)

    # Now our query handler
    @staticmethod
    def handle_query(
        ctx: click.Context,
        model: str,
        local: bool,
        raw: bool,
        temperature: float,
        chat: bool,
        append: str,
        prepend: str,
        query_input: str,
    ) -> None:
        from conduit.apps.cli.query.query_function import (
            CLIQueryFunctionInputs,
            CLIQueryFunctionProtocol,
        )

        printer: Printer = ctx.obj["printer"]

        inputs = CLIQueryFunctionInputs(
            query_input=query_input,
            printer=ctx.obj["printer"],
            context=prepend,
            append=append,
            system_message=ctx.obj["system_message"],
            name=ctx.obj["name"],
            cache=not local,
            local=local,
            preferred_model=model,
            verbose=ctx.obj["verbosity"],
            include_history=chat,
            temperature=temperature,
        )

        query_function: CLIQueryFunctionProtocol = ctx.obj["query_function"]
        response = query_function(inputs)
        if raw:
            printer.print_raw(response.content)
        else:
            printer.print_markdown(response.content)
