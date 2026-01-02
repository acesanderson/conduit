from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, override
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, model_validator, Field

from conduit.config import settings
from conduit.domain.message.message import MessageUnion, Message, ToolCall
from conduit.domain.message.role import Role
from conduit.domain.exceptions.exceptions import ConversationError
from conduit.domain.conversation.session import Session

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


class ConversationState(Enum):
    GENERATE = "generate"
    EXECUTE = "execute"
    TERMINATE = "terminate"
    INCOMPLETE = "incomplete"


class Conversation(BaseModel):
    """
    A Conversation is a sequence of Messages with metadata and helper methods.
    This a runtime object, not a persistence object. Conversation orchestrates the Session and Messages, with Session happening under the hood.
    A Conversation can be generated entirely from the message ID of a leaf Message.
    ONLY use the 'add' method to append new messages to ensure validation rules.
    Conversations are the primary interface for agents and chat-based LLM interactions, and as such, are where Messages are given their session and predecessor metadata.
    """

    topic: str = "Untitled"  # Note: for most applications, we auto label the Session, not the Conversation
    messages: list[MessageUnion] = Field(default_factory=list)
    session: Session | None = Field(default=None, repr=False)
    leaf: str | None = Field(default=None, repr=False)

    # Initialization Hooks
    @override
    def model_post_init(self, __context: object):
        """
        Bootstrap the session if initialized with a list of messages.
        """
        if self.messages:
            last_msg = self.messages[-1]
            self.leaf = last_msg.message_id

            # Check for viral session_id on the last message
            seed_id = getattr(last_msg, "session_id", None)

            self.initialize_session(
                leaf=self.leaf, session_id=seed_id, initial_messages=self.messages
            )

    def initialize_session(
        self,
        leaf: str,
        session_id: str | None = None,
        initial_messages: Sequence[MessageUnion] | None = None,
    ) -> None:
        """
        Lazily creates the Session object.
        """
        if self.session is None:
            # 1. Resolve ID (Viral > Random)
            final_id = session_id or f"session_{uuid.uuid4()}"

            # 2. Build initial dict
            msgs = initial_messages or []
            msg_dict = {m.message_id: m for m in msgs}

            # 3. Create Session
            self.session = Session(
                session_id=final_id, message_dict=msg_dict, leaf=leaf
            )

    @model_validator(mode="after")
    def validate_messages(self):
        messages = self.messages
        system_messages = [m for m in messages if m.role == Role.SYSTEM]

        if len(system_messages) > 1:
            raise ConversationError(
                "Multiple system messages found in the conversation."
            )
        elif len(system_messages) == 1:
            self.ensure_system_message(system_messages[0].content)
        return self

    # Methods to manipulate conversation
    def add(self, message: Message) -> None:
        """
        Append a new message to the conversation.
        Intercepts the addition to bootstrap the Session if needed.
        Enforces validation rules on message addition.
        1. Validates that system messages are only first, and no consecutive same-role messages
        2. Bootstraps the Session if not already initialized
        3. Updates the Session state with the new message
        4. Populates message metadata (predecessor_id, session_id)
        5. Updates the Conversation view (messages list, leaf pointer)
        """
        # --- 1. Validation First ---
        if len(self.messages) > 0:
            if message.role == Role.SYSTEM:
                raise ConversationError(
                    "System messages can only be the first message."
                )

            if message.role != Role.TOOL and message.role == self.messages[-1].role:
                raise ConversationError(
                    f"Cannot add two consecutive messages with the same role: {message.role.value}."
                )

        # --- 2. Bootstrap Session (Lazy Load) ---
        if self.session is None:
            seed_id = getattr(message, "session_id", None)
            # Initialize with THIS message
            self.initialize_session(
                leaf=message.message_id, session_id=seed_id, initial_messages=[message]
            )

        # --- 3. Update State ---
        assert self.session is not None
        self.session.register(message)

        # --- 4. Message metadata
        ## Lots of assertions to ensure integrity
        if self.last:
            assert self.last.predecessor_id == self.leaf
        predecessor_id = self.leaf
        assert self.session.session_id is not None, (
            "Session ID should be set after initialization."
        )
        if self.last:
            assert self.session.session_id == self.last.session_id, (
                "Session ID mismatch between Conversation and last Message."
            )
        session_id = self.session.session_id
        # Populate message metadata
        message.predecessor_id = predecessor_id
        message.session_id = session_id

        # --- 5. Update View ---
        self.messages.append(message)
        self.leaf = message.message_id

    def wipe(self) -> None:
        self.messages = []

    def prune(self, keep: int = 10) -> None:
        if len(self.messages) > keep:
            self.messages = self.messages[-keep:]

    def tokens(self, model_name: str) -> int:
        raise NotImplementedError(
            "Token counting not yet implemented for Conversation."
        )

    def ensure_system_message(
        self, system_content: str = settings.system_prompt
    ) -> None:
        from conduit.domain.message.message import SystemMessage

        system_message = SystemMessage(content=system_content)
        self.system = system_message

    def label(self, topic: str) -> None:
        """
        Automatic labeling happens to Session, not Conversation, but there are times when you might want to label a Conversation directly, for persistence. This stores a topic on the Message leaf metadata.
        """
        self.topic = topic
        self.leaf.metadata["topic"] = topic

    def branch(self, pointer: int | str) -> Conversation:
        """
        Create a new Conversation branch from a given message index or message ID.
        The new Conversation will share the same Session but have its own message list starting from the specified pointer.
        """
        if isinstance(pointer, int):
            if pointer < 0 or pointer >= len(self.messages):
                raise ConversationError("Pointer index out of range.")
            branch_messages = self.messages[: pointer + 1]
        elif isinstance(pointer, str):
            index = next(
                (i for i, m in enumerate(self.messages) if m.message_id == pointer), -1
            )
            if index == -1:
                raise ConversationError("Message ID not found in conversation.")
            branch_messages = self.messages[: index + 1]
        else:
            raise ConversationError(
                "Pointer must be an integer index or a message ID string."
            )

        branch_conversation = Conversation(
            topic=self.topic,
            messages=branch_messages,
            session=self.session,
        )
        return branch_conversation

    # Properties
    @property
    def last(self) -> Message | None:
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def system(self) -> Message | None:
        system_messages = [m for m in self.messages if m.role == Role.SYSTEM]
        if len(system_messages) == 1:
            return system_messages[0]
        elif len(system_messages) > 1:
            self.messages = [m for m in self.messages if m.role != Role.SYSTEM]
            self.messages.insert(0, system_messages[0])
            return system_messages[0]
        elif len(system_messages) == 0:
            return None

    @system.setter
    def system(self, message: Message):
        if message.role != Role.SYSTEM:
            raise ConversationError("Only system messages can be assigned to system.")

        existing_system = self.system
        if existing_system:
            index = self.messages.index(existing_system)
            self.messages[index] = message
        else:
            self.messages.insert(0, message)

        # Sync to session
        if self.session:
            self.session.register(message)

    @property
    def content(self) -> str:
        if self.last:
            return str(self.last.content)
        return ""

    @property
    def roles(self) -> str:
        roles_string = ""
        for message in self.messages:
            roles_string += message.role.value[0].upper()
        return roles_string

    @property
    def tool_calls(self) -> list[ToolCall]:
        if self.last:
            if self.last.tool_calls:
                return self.last.tool_calls
            else:
                raise ConversationError("Last message has no tool calls.")
        else:
            raise ConversationError(
                "No messages in conversation; cannot get tool calls."
            )

    @property
    def state(self) -> ConversationState:
        if not self.last:
            return ConversationState.INCOMPLETE
        match self.last.role:
            case Role.USER:
                return ConversationState.GENERATE
            case Role.TOOL:
                return ConversationState.GENERATE
            case Role.ASSISTANT:
                if self.last.tool_calls:
                    return ConversationState.EXECUTE
                else:
                    return ConversationState.TERMINATE
            case Role.SYSTEM:
                return ConversationState.INCOMPLETE

    # Display Methods
    @override
    def __str__(self) -> str:
        output = ""
        for message in self.messages:
            output += f"{message.role.value.upper()}: {message.content}\n"
        return output.strip()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        for message in self.messages:
            yield from message.__rich_console__(console, options)
        return

    def pretty_print(self) -> None:
        from rich.console import Console

        console = Console()
        console.print(self)

    def print_history(self, max_messages: int = 100) -> None:
        from rich.console import Console

        console = Console()
        truncated_messages = []
        for message in self.messages[-max_messages:]:
            truncated_content = str(message.content)
            truncated_content = " ".join(truncated_content.split())
            truncated_content = " ".join(truncated_content.split("  "))
            truncated_content = truncated_content.replace("**", "").replace("*", "")
            truncated_content = truncated_content.replace("`", "").replace("```", "")
            truncated_content = truncated_content.replace("_", "")
            truncated_content = truncated_content.replace("#", "")
            if len(truncated_content) > 120:
                truncated_content = truncated_content[:117] + "..."
            truncated_message = message.model_copy(
                update={"content": truncated_content}
            )
            truncated_messages.append(truncated_message)
        truncated_conversation = Conversation(
            topic=self.topic,
            messages=truncated_messages,
        )
        console.print(truncated_conversation)
