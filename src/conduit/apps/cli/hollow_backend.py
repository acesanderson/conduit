import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol
from collections.abc import Callable
from uuid import uuid4

# --- 1. Core Domain Models (The "What") ---


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Event:
    """An atomic item in the transcript (Spec 1.1)."""

    id: str = field(default_factory=lambda: str(uuid4()))
    role: Role = Role.USER
    content: str = ""

    # Branching metadata (Spec 1.3)
    branch_id: int = 0
    total_branches: int = 1

    # Properties for rendering
    is_expandable: bool = False
    is_collapsed: bool = False


@dataclass
class Conversation:
    """Immutable record of the event stream (Spec 1.2)."""

    id: str
    events: list[Event]


# --- 2. Application State (The "Truth") ---


class AppMode(Enum):
    """(Spec 2.0)"""

    NORMAL = auto()  # Navigation
    INSERT = auto()  # Typing
    COMMAND = auto()  # ':' overlay


@dataclass
class UIState:
    """Transient state for the TUI (Spec 10.12)."""

    mode: AppMode = AppMode.NORMAL

    # Navigation
    selected_event_id: str | None = None

    # Input Buffers
    main_input_buffer: str = ""
    command_buffer: str = ""

    # Visual flags
    show_help: bool = False
    is_streaming: bool = False


# --- 3. The Controller Interface (The "How") ---


class ControllerProtocol(Protocol):
    """
    The strict boundary between UI and Engine.
    The UI calls these methods; these methods mutate State.
    """

    state: UIState
    conversation: Conversation

    def register_redraw_callback(self, cb: Callable[[], None]):
        """So the engine can tell the UI to repaint (Spec 6.2)."""
        ...

    # -- Actions --

    def set_mode(self, mode: AppMode): ...

    def move_selection(self, direction: int):
        """Move up/down (-1/+1). Constraints: bounds check."""
        ...

    def toggle_expand(self):
        """Expand/collapse selected event."""
        ...

    def switch_branch(self, direction: int):
        """Cycle horizontal branches for selected event."""
        ...

    async def submit_input(self, text: str):
        """
        1. Commit user message.
        2. Start async assistant generation (streaming).
        """
        ...


# --- 4. The Mock Implementation (For Vibe Coding) ---


class MockController:
    """
    A functional mock to drive the TUI during development.
    Simulates streaming and branching.
    """

    def __init__(self):
        self.redraw_cb = lambda: None

        # Seed some fake data
        e1 = Event(role=Role.USER, content="Hello, system.")
        e2 = Event(role=Role.ASSISTANT, content="Greetings. Ready.", total_branches=2)

        self.conversation = Conversation(id="conv_1", events=[e1, e2])
        self.state = UIState(selected_event_id=e2.id)

    def register_redraw_callback(self, cb):
        self.redraw_cb = cb

    def _update(self):
        self.redraw_cb()

    def set_mode(self, mode: AppMode):
        self.state.mode = mode
        self._update()

    def move_selection(self, direction: int):
        # (Mock logic to find current index and move up/down)
        ids = [e.id for e in self.conversation.events]
        try:
            curr_idx = ids.index(self.state.selected_event_id)
            new_idx = max(0, min(len(ids) - 1, curr_idx + direction))
            self.state.selected_event_id = ids[new_idx]
            self._update()
        except ValueError:
            pass

    async def submit_input(self, text: str):
        # 1. Add User Message
        user_msg = Event(role=Role.USER, content=text)
        self.conversation.events.append(user_msg)
        self.state.selected_event_id = user_msg.id
        self.state.main_input_buffer = ""  # Clear input
        self.set_mode(AppMode.NORMAL)
        self._update()

        # 2. Simulate Assistant Streaming
        asst_msg = Event(role=Role.ASSISTANT, content="")
        self.conversation.events.append(asst_msg)
        self.state.selected_event_id = asst_msg.id
        self.state.is_streaming = True

        # Fake Stream Loop
        full_response = "This is a simulated streaming response based on your input."
        for word in full_response.split():
            await asyncio.sleep(0.1)  # Fake network lag
            asst_msg.content += word + " "
            self._update()  # Trigger UI redraw

        self.state.is_streaming = False
        self._update()
