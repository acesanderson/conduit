"""
Our base handlers.

To extend, mixin your own class with this one on ChatApp; or inherit and override.
Handlers must:

(1) be decorated with @command to be registered.
(2) have docstrings for help display.
(3) avoid displaying output whenever possible; instead return strings to be printed.
"""

from conduit.chat.command import command


class Handlers:
    """
    Mixin providing command handler methods for chat applications.

    Each handler method corresponds to a slash-prefixed chat command (e.g., /exit, /help, /set model).
    Methods avoid direct console output where possible, returning strings instead for flexible display handling.

    Requires mixed-in class to provide:
    - self.console: Rich Console instance for output
    - self.message_store: MessageStore for history operations (optional)
    - self.model: Model instance for LLM interactions
    - self.clipboard_image: Storage for image context (optional)
    """

    @command("exit", aliases=["quit", "q", "bye"])
    def command_exit(self):
        """
        Exit the chat.
        """
        exit()

    @command("help", aliases=["h", "?"])
    def command_help(self):
        """Display available commands."""
        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Command", style="green")
        table.add_column("Description", style="yellow")

        for cmd in self.get_all_commands():
            aliases_str = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            param_str = {0: "", 1: " <arg>", "multi": " <args...>"}[cmd.param_count]

            command_name = f"/{cmd.name}{param_str}{aliases_str}"
            description = cmd.description.strip()

            table.add_row(command_name, description)

        # Return None and print directly since Rich tables can't be returned as strings
        self.console.print(table)
        return None

    @command("wipe")
    def command_wipe(self):
        """
        Clear the message history.
        """
        if self.message_store:
            self.message_store.clear()
            return "[green]Message history cleared.[/green]"
        else:
            return "[red]No message store available.[/red]"

    @command("clear", aliases=["cls"])
    def command_clear(self):
        """
        Clear the screen.
        """
        self.console.clear()

    @command("show log level", aliases=["log", "log level"])
    def command_show_log_level(self):
        raise NotImplementedError("Log level not implemented yet.")

    @command("set log level", param_count=1, aliases=["set log"])
    def command_set_log_level(self, param: str):
        raise NotImplementedError("Log level not implemented yet.")

    @command("show history", aliases=["history", "hi"])
    def command_show_history(self):
        """
        Display the chat history.
        """
        if self.message_store:
            self.message_store.view_history()

    @command("show models", aliases=["models", "ms"])
    def command_show_models(self):
        """
        Display available models.
        """
        from conduit.model.models.modelstore import ModelStore

        ms = ModelStore()
        ms.display()

    @command("show model", aliases=["model", "m"])
    def command_show_model(self):
        """
        Display the current model.
        """
        return f"[green]Current model: {self.model.model}[/green]"

    @command("set model", param_count=1, aliases=["sm"])
    def command_set_model(self, param: str):
        """
        Set the current model.
        """
        try:
            self.model = Model(param)
            self.console.print(f"Set model to {param}", style="green")
        except ValueError:
            self.console.print(f"Invalid model: {param}", style="red")

    @command("paste image", aliases=["pi"])
    def command_paste_image(self):
        """
        Use this when you have an image in clipboard that you want to submit as context for LLM.
        This gets saved as self.clipboard_image.
        """
        from Conduit.message.imagemessage import ImageMessage
        import os

        if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
            self.console.print("Image paste not available over SSH.", style="red")
            return

        import warnings
        from PIL import ImageGrab
        import base64, io

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress PIL warnings
            image = ImageGrab.grabclipboard()

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            # Save for next query
            self.console.print("Image captured! What is your query?", style="green")
            user_input = self.console.input(
                "[bold green]Query about image[/bold green]: "
            )
            # Build our ImageMessage
            text_content = user_input
            image_content = img_base64
            mime_type = "image/png"
            role = "user"
            imagemessage = ImageMessage(
                role=role,
                text_content=text_content,
                image_content=image_content,
                mime_type=mime_type,
            )
            self.message_store.append(imagemessage)
            response = self.query_model([self.message_store.last()])
            self.console.print(response)

        else:
            self.console.print("No image detected.", style="red")

    @command("wipe image", aliases=["wi"])
    def command_wipe_image(self):
        """
        Delete image from memory.
        """
        if self.clipboard_image:
            self.clipboard_image = None
            return "[green]Image deleted.[/green]"
        else:
            return "[red]No image to delete.[/red]"
