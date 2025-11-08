from abc import ABC, abstractmethod


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
    def show_message(self, message: str, style: str = "info") -> None:
        """
        Display message to user
        """
        pass

    @abstractmethod
    def clear_screen(self) -> None:
        """
        Clear the screen
        """
        pass
