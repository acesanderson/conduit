"""
A Conversation wraps a list[Message] with extra metadata, validation, and helper methods.
It's also a core data object for persistence.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.message.message import Message
from conduit.domain.message.role import Role
from pydantic import BaseModel, Field, model_validator
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from conduit.domain.request.generation_params import GenerationParams
    from rich.console import Console, ConsoleOptions, RenderResult


class ConversationState(Enum):
    GENERATE = "generate"
    EXECUTE = "execute"
    TERMINATE = "terminate"
    INCOMPLETE = "incomplete"
    # --- Future: Context Engineering (Memory) ---
    # SUMMARIZE = "summarize"  # Trigger: Context > limit. Action: Compress history -> GENERATE
    # INJECT = "inject"        # Trigger: RAG/Search needed. Action: Retrieve & Insert Context -> GENERATE
    # FORGET = "forget"        # Trigger: Topic shift. Action: Prune irrelevant history -> GENERATE

    # --- Future: Agentic Workflow (Planning) ---
    # CLASSIFY = "classify"    # Trigger: Start of Conv. Action: Route to specific Model/Skill -> GENERATE
    # PLAN = "plan"            # Trigger: Complex task. Action: Generate Chain-of-Thought steps -> EXECUTE
    # DECOMPOSE = "decompose"  # Trigger: Multi-part query. Action: Split into sub-conversations -> GENERATE

    # --- Future: Quality Control (Reflection) ---
    # VALIDATE = "validate"    # Trigger: Post-GENERATE. Action: Syntax check (JSON/Code) -> TERMINATE or REFINE
    # REFINE = "refine"        # Trigger: Validation failed. Action: Self-correction prompt -> GENERATE
    # CRITIQUE = "critique"    # Trigger: High-quality mode. Action: "Review your answer" -> REFINE

    # --- Future: Safety & Human Interaction ---
    # CONFIRM = "confirm"      # Trigger: Sensitive tool call. Action: Pause flow, wait for user signal -> EXECUTE
    # AWAIT = "await"          # Trigger: Long-running async task. Action: Poll/Wait -> TERMINATE


class ConversationError(Exception):
    pass


class Conversation(BaseModel):
    topic: str = "Untitled"
    messages: list[Message] = []

    # Generated fields
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

    @model_validator(mode="before")
    def validate_messages(cls, values):
        """
        Ensure messages is a list of Message objects, and system message rules.
        """
        # Validate messages
        messages = values.get("messages", [])
        if not isinstance(messages, list):
            raise ConversationError("Messages must be a list of Message objects.")
        for message in messages:
            if not isinstance(message, Message):
                raise ConversationError(
                    "All items in messages must be instances of Message."
                )
        # If there is a system message, ensure that (1) it's first and (2) there's only one
        system_messages = [m for m in messages if m.role == "system"]
        if len(system_messages) > 1:
            raise ConversationError(
                "Multiple system messages found in the conversation."
            )
        elif len(system_messages) == 1:
            self.ensure_system_message()
        return values

    # Helper properties
    def add(self, message: Message) -> None:
        """
        Append a new message to the conversation.
        Use this INSTEAD of manipulating messages directly, since there are validation rules.
        """
        if len(self.messages) > 0:
            if message.role == Role.SYSTEM:
                raise ConversationError(
                    "System messages can only be the first message."
                )
            if message.role != Role.TOOL and message.role == self.messages[-1].role:
                # Exempt Tool messages from this rule, since there can be multiple tool calls in a row
                raise ConversationError(
                    f"Cannot add two consecutive messages with the same role: {message.role.value}."
                )

        self.messages.append(message)

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
            # Remove all system messages but the first
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
            # Replace existing system message
            index = self.messages.index(existing_system)
            self.messages[index] = message
        else:
            # Insert new system message at the start
            self.messages.insert(0, message)

    @property
    def content(self) -> str:
        """
        Content from last message; to get results from generation response, or capture user prompt.
        """
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

    def wipe(self) -> None:
        """
        Clear all messages.
        """
        self.messages = []

    def prune(self, keep: int = 10) -> None:
        """
        Keep only the last `keep` messages.
        """
        if len(self.messages) > keep:
            self.messages = self.messages[-keep:]

    def tokens(self, model_name: str) -> int:
        raise NotImplementedError(
            "Token counting not yet implemented for Conversation."
        )

    def ensure_system_message(
        self, system_content: str = settings.system_prompt
    ) -> None:
        """
        Convenience method: converts string to SystemMessage and ensures it's first.
        """
        from conduit.domain.message.message import SystemMessage

        system_message = SystemMessage(content=system_content)
        self.system = system_message

    # Display dunders
    @override
    def __str__(self) -> str:
        """
        Full string representation of the conversation history.
        """
        output = ""
        for message in self.messages:
            output += f"{message.role.value.upper()}: {message.content}\n"
        return output.strip()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """
        Every message object has a __rich_console__ method, so we can leverage that here.
        Combine those renderables into one for the conversation.
        """
        for message in self.messages:
            yield from message.__rich_console__(console, options)
        return


if __name__ == "__main__":
    # Simple test of conversation display, first create a dummy conversation
    from conduit.domain.message.message import UserMessage, AssistantMessage

    conv = Conversation(topic="Test Conversation")
    conv.ensure_system_message("You are a helpful assistant.")
    conv.add(UserMessage(content="Hello, how are you?"))
    conv.add(
        AssistantMessage(
            content="I'm doing well, thank you! How can I assist you today?"
        )
    )
    print(conv)
    from rich.console import Console

    console = Console()
    console.print(conv)
