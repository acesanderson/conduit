from enum import Enum, auto


class UICommand(Enum):
    """
    Enumeration of UI commands for the chat interface.

    Defines high-level UI operations that can be triggered by commands, distinct from
    the command execution layer. These represent semantic actions that modify UI state
    or behavior rather than direct command implementations.
    """

    CLEAR_SCREEN = auto()
    CLEAR_HISTORY_FILE = auto()
    EXIT = auto()
