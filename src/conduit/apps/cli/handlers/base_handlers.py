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
import json
import sys
import asyncio
import click

if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    from conduit.apps.cli.utils.printer import Printer
    from conduit.domain.message.message import UserMessage, Message, ImageContent, AudioContent
    from conduit.storage.repository.protocol import AsyncSessionRepository
    from conduit.apps.cli.query.query_function import CLIQueryFunctionProtocol
    from uuid import UUID

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VERBOSITY = settings.default_verbosity


def _save_response(response: object, path: str) -> None:
    """
    Write response output to a file. Priority: audio > images > text.
    response is a Conversation; response.last is the AssistantMessage.
    """
    import base64
    from pathlib import Path

    last = response.last
    _audio = getattr(last, "audio", None)
    if _audio and isinstance(getattr(_audio, "data", None), str):
        data = base64.b64decode(last.audio.data)
        Path(path).write_bytes(data)
        logger.info("Saved audio response to %s", path)
    elif getattr(last, "images", None):
        data = base64.b64decode(last.images[0].b64_json)
        Path(path).write_bytes(data)
        logger.info("Saved image response to %s", path)
    else:
        Path(path).write_text(str(last))
        logger.info("Saved text response to %s", path)


def _play_audio(path: str) -> None:
    """Play an audio file via pydub. Requires simpleaudio or pyaudio as pydub backend."""
    from pydub import AudioSegment
    from pydub.playback import play

    logger.info("Playing audio from %s", path)
    audio = AudioSegment.from_file(path)
    play(audio)


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

    # Handlers
    @staticmethod
    def handle_history(
        repository: AsyncSessionRepository,
        session_id: str,
        printer: Printer,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        View message history and exit.
        """
        logger.info("Viewing message history...")

        # Async fetch via the passed loop
        session = loop.run_until_complete(repository.get_session(session_id))

        if not session:
            printer.print_pretty("[yellow]Conversation not found.[/yellow]")
            sys.exit(1)

        conversation = session.conversation
        conversation.print_history()
        sys.exit()

    @staticmethod
    def handle_wipe(
        printer: Printer,
        repository: AsyncSessionRepository,
        session_id: str,
        loop: asyncio.AbstractEventLoop,
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
            # Async delete via the passed loop
            loop.run_until_complete(repository.delete_session(session_id))
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
        last_message: Message = conversation.last
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

    @staticmethod
    def handle_query(
        query_input: str,
        model: str,
        local: bool,
        raw: bool,
        temperature: float | None,
        chat: bool,
        append: str | None,
        verbosity: Verbosity,
        # Injected Dependencies
        printer: Printer,
        query_function: CLIQueryFunctionProtocol,
        stdin: str | None,
        system_message: str = "",
        project_name: str = "",
        search: bool = False,
        citations: bool = False,
        image_path: str | None = None,
        image_content: ImageContent | None = None,
        audio_path: str | None = None,
        audio_content: AudioContent | None = None,
        save: str | None = None,
        play: bool = False,
        persist: bool = False,
    ) -> None:
        """
        Here we resolve all inputs for flat input to the query function.
        Dependencies are injected explicitly, removing the Click context coupling.
        """
        from conduit.apps.cli.query.query_function import CLIQueryFunctionInputs

        # 1. Normalize Context (stdin)
        # Handle cases where stdin might be an empty string or whitespace
        context_text = stdin if isinstance(stdin, str) and stdin.strip() else ""

        # 2. Build Inputs
        client_params = {"return_citations": True} if citations else {}

        inputs = CLIQueryFunctionInputs(
            query_input=query_input,
            printer=printer,
            context=context_text,
            append=append or "",
            system_message=system_message,
            project_name=project_name,
            search=search,
            cache=not local,
            local=local,
            preferred_model=model,  # Model is already resolved by the Command layer
            verbose=verbosity,
            include_history=chat,
            temperature=temperature,
            client_params=client_params,
            image_path=image_path,
            image_content=image_content,
            audio_path=audio_path,
            audio_content=audio_content,
            ephemeral=not persist,
        )

        # 3. Execute
        response = query_function(inputs)

        # 4. Resolve effective save path
        from datetime import datetime
        last = response.last
        effective_save = save

        _audio = getattr(last, "audio", None)
        if _audio and isinstance(getattr(_audio, "data", None), str) and not save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            effective_save = f"/tmp/conduit_audio_{ts}.{last.audio.format}"
            logger.info("Auto-saving audio to %s", effective_save)

        # 5. Save, play, or display
        if effective_save:
            _save_response(response, effective_save)
        else:
            if raw:
                printer.print_raw(response.content)
            else:
                printer.print_markdown(response.content)

        if play:
            if effective_save and getattr(response.last, "audio", None):
                _play_audio(effective_save)
            else:
                logger.warning("--play set but no audio response to play")

        # 6. Citations
        BaseHandlers.handle_citations(response, citations=citations, raw=raw, printer=printer)

    @staticmethod
    def handle_citations(
        response: object,
        citations: bool,
        raw: bool,
        printer: Printer,
    ) -> None:
        """
        Print citations after a query response.
        All warnings go to stderr via printer.print_err.
        Raw JSON goes to stdout via click.echo. Formatted list goes to printer.print_citations.
        """
        if not citations:
            return

        # Provider check — only Perplexity populates citations
        # Try GenerationResponse shape (.message) first, fall back to Conversation shape (.last)
        message = getattr(response, "message", None) or getattr(response, "last", None)
        metadata: dict = getattr(message, "metadata", {}) or {}
        provider: str | None = metadata.get("provider")

        if provider != "perplexity":
            logger.warning(
                "--citations requested but provider is %r, not 'perplexity'", provider
            )
            printer.print_err(
                "[red]--citations is only supported for Perplexity models "
                "(sonar, sonar-pro). No citations available.[/red]"
            )
            return

        citations_list: list[dict] = metadata.get("citations", [])

        if not citations_list:
            logger.debug("--citations: no citations returned by model")
            printer.print_err(
                "[yellow]No citations were returned by the model.[/yellow]"
            )
            return

        logger.debug("--citations: printing %d citations", len(citations_list))

        if raw:
            click.echo(json.dumps(citations_list))
        else:
            printer.print_citations(citations_list)
