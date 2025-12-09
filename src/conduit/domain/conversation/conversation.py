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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.request.generation_params import GenerationParams


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
        elif system_content is None:
            return
        else:
            content: str = system_content or settings.system_prompt
            from conduit.domain.message.message import SystemMessage

            system_message = SystemMessage(content=content)
            self.messages.insert(0, system_message)
