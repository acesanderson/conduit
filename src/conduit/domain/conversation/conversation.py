"""
A Conversation wraps a Sequence[Message] with extra metadata, validation, and helper methods.
It's also a core data object for persistence.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.message.message import MessageUnion, Message
from conduit.domain.message.role import Role
from pydantic import BaseModel, Field, model_validator
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, override
from collections.abc import Sequence

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
    messages: Sequence[MessageUnion] = []

    # Generated fields
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

    @model_validator(mode="after")
    def validate_messages(self):
        """
        Runs AFTER Pydantic has parsed the dict into a Conversation object.
        'self' is the Conversation instance.
        """
        messages = self.messages
        system_messages = [m for m in messages if m.role == Role.SYSTEM]

        if len(system_messages) > 1:
            raise ConversationError(
                "Multiple system messages found in the conversation."
            )
        elif len(system_messages) == 1:
            self.ensure_system_message(system_messages[0].content)
        return self

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

    def print_history(self, max_messages: int = 100) -> None:
        """
        Print the last `max_messages` messages to the console using rich.
        Unlike __rich_console__, this pretty prints each message but ONLY the first 90 characters for each text block.
        Generate a new Conversation with truncated messages and then use rich with __rich_console__ to print it.
        """
        from rich.console import Console

        console = Console()
        truncated_messages = []
        for message in self.messages[-max_messages:]:
            truncated_content = str(message.content)
            # Remove multi spaces and newlines for truncation
            truncated_content = " ".join(truncated_content.split())
            # Remove more than one space in a sequence
            truncated_content = " ".join(truncated_content.split("  "))
            # Remove markdown formatting for truncation
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
            conversation_id=self.conversation_id,
            timestamp=self.timestamp,
        )
        console.print(truncated_conversation)


if __name__ == "__main__":
    # Simple test of conversation display, first create a dummy conversation
    from conduit.domain.message.message import UserMessage, AssistantMessage

    conv = Conversation(topic="Test Conversation")
    conv.ensure_system_message("""You are a helpful assistant. Please assist the user with the following ten things:
        1. Be concise.
        2. Be accurate.
        3. Be polite.
        4. Provide examples when relevant.
        5. Use proper grammar.
        6. Avoid jargon.
        7. Stay on topic.
        8. Ask clarifying questions if needed.
        9. Summarize key points.
        10. End with a friendly closing.""")
    conv.add(
        UserMessage(
            content="""Please answer these ten questions:
        1. What is the capital of France?
        2. Who wrote 'To Kill a Mockingbird'?
        3. What is the largest planet in our solar system?
        4. How many continents are there on Earth?
        5. What is the boiling point of water?
        6. Who painted the Mona Lisa?
        7. What is the smallest prime number?
        8. What year did the Titanic sink?
        9. Who is known as the 'Father of Computers'?
        10. What is the chemical symbol for gold?"""
        )
    )
    conv.add(
        AssistantMessage(
            content="I'm doing well, thank you! How can I assist you today?"
        )
    )
    print(conv)
    from rich.console import Console

    console = Console()
    conv.print_history()
