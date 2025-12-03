"""
A Conversation wraps a list[Message] with extra metadata, validation, and helper methods.
It's also a core data object for persistence.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.message.message import Message
from conduit.domain.message.role import Role
from pydantic import BaseModel, Field
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.request.generation_params import GenerationParams


class ConversationState(Enum):
    GENERATE = "generate"
    EXECUTE = "execute"
    TERMINATE = "terminate"
    INCOMPLETE = "incomplete"


class ConversationError(Exception):
    pass


class Conversation(BaseModel):
    messages: list[Message] = []
    metadata: GenerationParams | None = None

    # Generated fields
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

    # Helper properties
    @property
    def last(self) -> Message | None:
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def system(self) -> Message | None:
        for message in self.messages:
            if message.role == "system":
                return message
        return None

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

    def tokens(self, model_name: str) -> int:
        raise NotImplementedError(
            "Token counting not yet implemented for Conversation."
        )

    # Methods
    def add_message(self, message: Message):
        """
        Add a message to the conversation after validating it.
        """
        if self.last and self.last.role == message.role:
            raise ValueError("Cannot add two consecutive messages with the same role.")
        self.messages.append(message)

    def ensure_system_message(self, system_content: str | None = None):
        """
        Ensure that a system message exists in the conversation.
        Ensure that only one system message exists.
        Ensure that the system message is the first message.
        If it doesn't exist, create one with the provided content or a default (settings.SYSTEM_PROMPT).
        """
        system_messages = [m for m in self.messages if m.role == "system"]
        if len(system_messages) > 1:
            raise ValueError("Multiple system messages found in the conversation.")
        elif len(system_messages) == 1:
            system_message = system_messages[0]
            if self.messages[0] != system_message:
                self.messages.remove(system_message)
                self.messages.insert(0, system_message)
        else:
            content: str = system_content or settings.system_prompt
            from conduit.domain.message.message import SystemMessage

            system_message = SystemMessage(content=content)
            self.messages.insert(0, system_message)
