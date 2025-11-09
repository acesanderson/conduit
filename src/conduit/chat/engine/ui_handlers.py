from conduit.chat.ui.ui_command import UICommand
from conduit.chat.engine.command import command, CommandResult


class UICommandHandlers:
    """
    Mixin providing UI command handlers for chat applications.

    Implements slash-prefixed commands that return UICommand enums to trigger
    high-level UI operations (screen clearing, history management, application exit)
    distinct from content-focused command handlers. Methods must be decorated with
    @command and return UICommand enum values for the caller to execute.

    Requires mixed-in class to be compatible with CommandDispatcher infrastructure.
    """

    @command("clear", aliases=["cls"])
    def clear(self) -> CommandResult:
        """
        Clear the screen.
        """
        return UICommand.CLEAR_SCREEN

    @command("clear history", aliases=["ch"])
    def clear_history(self) -> CommandResult:
        """
        Clear the chat history.
        """
        return UICommand.CLEAR_HISTORY_FILE

    @command("exit", aliases=["quit", "q", "bye"])
    def exit(self) -> CommandResult:
        """
        Exit the chat.
        """
        return UICommand.EXIT
