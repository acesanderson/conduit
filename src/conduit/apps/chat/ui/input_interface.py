from abc import ABC, abstractmethod
from conduit.chat.ui.ui_command import UICommand
from rich.console import RenderableType


class InputInterface(ABC):
    """
    Abstract interface for user input
    """

    @abstractmethod
    def get_input(self, prompt: str = ">> ") -> str:
        """
        Get user input. May be single or multi-line.
        """
        pass

    @abstractmethod
    def show_message(self, message: RenderableType, style: str = "info") -> None:
        """
        Display message to user
        """
        pass

    # UI commands
    @abstractmethod
    def execute_ui_command(self, command: UICommand) -> None:
        """
        Execute a UI command
        """
        pass

    @abstractmethod
    def clear_screen(self) -> None:
        """
        Clear the screen
        """
        pass

    @abstractmethod
    def clear_history_file(self) -> None:
        """
        Clear the persistent history file
        """
        pass

    @abstractmethod
    def exit(self) -> None:
        """
        Exit the application
        """
        pass
