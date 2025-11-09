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
from conduit.chat.ui.ui_command import UICommand


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

    @command("exit", aliases=["quit", "q", "bye"])
    def exit(self) -> CommandResult:
        """
        Exit the chat.
        """
        return UICommand.EXIT

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

        # Return None and print directly since Rich tables can't be returned as strings
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
            return "[red]No message store available.[/red]"

    @command("clear", aliases=["cls"])
    def clear(self) -> CommandResult:
        """
        Clear the screen.
        """
        return UICommand.CLEAR_SCREEN

    @command("show log level", aliases=["log", "log level"])
    def show_log_level(self) -> CommandResult:
        raise NotImplementedError("Log level not implemented yet.")

    @command("set log level", param_count=1, aliases=["set log"])
    def set_log_level(self, param: str) -> CommandResult:
        raise NotImplementedError("Log level not implemented yet.")

    @command("show history", aliases=["history", "hi"])
    def show_history(self):
        """
        Display the chat history.
        """
        if self.message_store:
            self.message_store.view_history()

    @command("show models", aliases=["models", "ms"])
    def show_models(self) -> CommandResult:
        """
        Display available models.
        """
        from conduit.model.models.modelstore import ModelStore

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
        from conduit.model.model import Model

        try:
            self.model = Model(param)
            return f"[green]Set model to {param}[/green']"
        except ValueError:
            return f"[red]Invalid model: {param}[/red]"

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
