"""
Our base handlers.

To extend, mixin your own class with this one on ChatApp; or inherit and override.
Handlers must:

(1) be decorated with @command to be registered.
(2) have docstrings for help display.
(3) avoid displaying output whenever possible; instead return strings* to be printed**.
(4) raise NotImplementedError for unimplemented commands.
(5) Lazy load imports within methods to avoid circular dependencies.

* Technically this is RenderableType from Rich, so in addition to strings, you can return Rich objects like Tables, Panels, Markdown etc.
** If UI commands are returned (like CLEAR_SCREEN), the caller is expected to handle them appropriately.
"""

from conduit.chat.engine.command import command, CommandResult
from conduit.chat.engine.exceptions import CommandError


class CommandHandlers:
    """
    Mixin providing command handler methods for chat applications.

    Each handler method corresponds to a slash-prefixed chat command (e.g., /exit, /help, /set model).
    Methods avoid direct console output where possible, returning strings instead for flexible display handling.

    Requires mixed-in class to provide:
    - self.message_store: MessageStore for history operations (optional)
    - self.model: Model instance for LLM interactions
    - self.clipboard_image: Storage for image context (optional)
    """

    @command("help", aliases=["h", "?"])
    def help(self) -> CommandResult:
        """Display available commands."""
        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Command", style="green")
        table.add_column("Description", style="yellow")

        for cmd in self.get_all_commands():
            aliases_str = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            param_str = {0: "", 1: " <arg>", "multi": " <args...>"}[cmd.param_count]

            name = f"/{cmd.name}{param_str}{aliases_str}"
            description = cmd.description.strip()

            table.add_row(name, description)

        return table

    @command("wipe")
    def wipe(self) -> CommandResult:
        """
        Clear the message history.
        """
        if self.message_store:
            self.message_store.clear()
            return "[green]Message history cleared.[/green]"
        else:
            raise CommandError("No message store available to wipe history.")

    @command("show history", aliases=["history", "hi"])
    def show_history(self):
        """
        Display the chat history.
        """
        if self.message_store:
            self.message_store.view_history()
        else:
            raise CommandError("No message store available to show history.")

    @command("show models", aliases=["models", "ms"])
    def show_models(self) -> CommandResult:
        """
        Display available models.
        """
        from conduit.core.model.models.modelstore import ModelStore

        ms = ModelStore()
        renderable_columns = ms._generate_renderable_model_list()
        return renderable_columns

    @command("show model", aliases=["model", "m"])
    def show_model(self) -> CommandResult:
        """
        Display the current model.
        """
        return f"[green]Current model: {self.model.model}[/green]"

    @command("set model", param_count=1, aliases=["sm"])
    def set_model(self, param: str) -> CommandResult:
        """
        Set the current model.
        """
        from conduit.core.model.model_sync import ModelSync

        try:
            self.model = Model(param)
            return f"[green]Set model to {param}[/green]"
        except ValueError:
            raise CommandError(f"Invalid model: {param}")

    def _latest_response(self) -> str:
        """
        Helper to get the latest response from message store.
        """
        if self.message_store:
            for message in reversed(self.message_store.messages):
                if message.role == "assistant":
                    return str(message.content)
        else:
            raise CommandError("No message store available to retrieve responses.")

    @command("clip")
    def clip(self) -> CommandResult:
        import pyperclip

        """
        Copy the latest LLM response to clipboard.
        """
        latest_response = self._latest_response()
        if latest_response:
            pyperclip.copy(latest_response)
            return "[green]Latest response copied to clipboard.[/green]"
        else:
            raise CommandError("No assistant response found to copy.")

    @command("note", 1, aliases=["no", "obsidian"])
    def note(self, name: str) -> CommandResult:
        """
        Save the latest LLM response as a note in Obsidian.
        """
        from pathlib import Path
        import os

        latest_response = self._latest_response()
        if not latest_response:
            raise CommandError("No assistant response found to save as note.")

        obsidian_vault_path = os.getenv("OBSIDIAN_PATH")
        if not obsidian_vault_path:
            raise CommandError("OBSIDIAN_PATH environment variable not set.")

        obsidian_vault_dir = Path(obsidian_vault_path)
        note_path = obsidian_vault_dir / f"{name}.md"
        if note_path.exists():
            raise CommandError(f"Note '{name}.md' already exists in Obsidian vault.")
        _ = note_path.write_text(latest_response, encoding="utf-8")
        return f"[green]Note saved to {note_path}[/green]"

    @command("tangent", aliases=["tan"])
    def tangent(self) -> CommandResult:
        """
        Start a tangent conversation, preserving current context but branching off.
        """
        # if self.message_store:
        #     self.message_store.start_tangent() # Implement tangent logic in MessageStore
        #     return "[green]Tangent conversation started.[/green]"
        # else:
        #     raise CommandError("No message store available to start tangent.")
        raise NotImplementedError("Tangent conversations not implemented yet.")

    @command("paste image", aliases=["pi"])
    def paste_image(self) -> CommandResult:
        """
        Use this when you have an image in clipboard that you want to submit as context for LLM.
        This gets saved as self.clipboard_image.
        """
        raise NotImplementedError("Image paste not implemented yet.")
        # from Conduit.message.imagemessage import ImageMessage
        # import os
        #
        # if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
        #     self.input_interface.show_message(
        #         "Image paste not available over SSH.", style="red"
        #     )
        #     return
        #
        # import warnings
        # from PIL import ImageGrab
        # import base64, io
        #
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")  # Suppress PIL warnings
        #     image = ImageGrab.grabclipboard()
        #
        # if image:
        #     buffer = io.BytesIO()
        #     image.save(buffer, format="PNG")
        #     img_base64 = base64.b64encode(buffer.getvalue()).decode()
        #     # Save for next query
        #     self.input_interface.show_message(
        #         "Image captured! What is your query?", style="green"
        #     )
        #     user_input = self.console.input(
        #         "[bold green]Query about image[/bold green]: "
        #     )
        #     # Build our ImageMessage
        #     text_content = user_input
        #     image_content = img_base64
        #     mime_type = "image/png"
        #     role = "user"
        #     imagemessage = ImageMessage(
        #         role=role,
        #         text_content=text_content,
        #         image_content=image_content,
        #         mime_type=mime_type,
        #     )
        #     self.message_store.append(imagemessage)
        #     response = self.query_model([self.message_store.last()])
        #     self.input_interface.show_message(response)
        #
        # else:
        #     self.input_interface.show_message("No image detected.", style="red")

    @command("wipe image", aliases=["wi"])
    def wipe_image(self) -> CommandResult:
        """
        Delete image from memory.
        """
        raise NotImplementedError("Image wipe not implemented yet.")
        # if self.clipboard_image:
        #     self.clipboard_image = None
        #     return "[green]Image deleted.[/green]"
        # else:
        #     return "[red]No image to delete.[/red]"
