from conduit.apps.chat.ui.input_interface import InputInterface
from conduit.apps.chat.ui.ui_command import UICommand
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from typing import override
import re

# Precompile regex pattern to detect Rich style tags
style_pattern = re.compile(r"\[/?[a-zA-Z0-9_ ]+\]")


class BasicInput(InputInterface):
    """
    Minimal input using Rich Console (current implementation)
    """

    def __init__(self, console: Console):
        self.console: Console = console

    @override
    async def get_input(self, prompt: str = ">> ") -> str:
        """
        Get user input from console with formatted prompt styling.
        """
        return self.console.input(f"[bold gold3]{prompt}[/bold gold3]")

    @override
    def show_message(self, message: RenderableType, style: str = "") -> None:
        """
        Display a message to the user via Rich console.
        Automatically convert string messages to Markdown for formatted output.
        """
        if isinstance(message, str):
            # If a style is explicitly provided, use it directly
            if style:
                self.console.print(message, style=style)
                return
            # If style tags are detected, print as-is
            if style_pattern.search(message):
                self.console.print(message)
                return
            # This is a plain string, convert to Markdown
            else:
                message = Markdown(message)
                self.console.print(message)
        # This is already a RenderableType (e.g., Markdown, Table, etc.)
        else:
            self.console.print(message)

    @override
    def execute_ui_command(self, command: UICommand) -> None:
        if command == UICommand.CLEAR_SCREEN:
            self.clear_screen()
        elif command == UICommand.CLEAR_HISTORY_FILE:
            self.clear_history_file()
        elif command == UICommand.EXIT:
            self.exit()
        else:
            raise NotImplementedError(
                f"UI command {command} not implemented in BasicInput."
            )

    @override
    def clear_screen(self) -> None:
        self.console.clear()

    @override
    def clear_history_file(self) -> None:
        self.console.print(
            "[yellow]Clearing history file is not supported in BasicInput.[/yellow]"
        )

    @override
    def exit(self) -> None:
        import sys

        sys.exit(0)
