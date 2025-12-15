"""
NOTE: Typing is a mess in python. We have two types for Message:
- Message: the base class for all messages, used for isinstance checks.
- MessageUnion: a discriminated union of all message types, used for parsing/serialization.

* Enums don't work as discriminators in pydantic, so we use a new field 'role_str' for that purpose.
"""

from __future__ import annotations
from conduit.domain.message.role import Role
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any, Annotated, override
import time
import uuid
import functools


# Multimodal display helpers
def display(image_content: str) -> None:
    """
    Display a base64-encoded image using chafa.
    Your mileage may vary depending on the terminal and chafa version.
    """
    import subprocess
    import base64
    import os

    try:
        image_data = base64.b64decode(image_content)
        cmd = ["chafa", "-"]

        # If in tmux or SSH, force text mode for consistency
        if (
            os.environ.get("TMUX")
            or os.environ.get("SSH_CLIENT")
            or os.environ.get("SSH_CONNECTION")
        ):
            cmd.extend(["--format", "symbols", "--symbols", "block"])
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.communicate(input=image_data)
    except Exception as e:
        print(f"Error displaying image: {e}")


def play(audio_content: str):
    """
    Play the audio from the base64 content (no file required).
    """
    from pydub import AudioSegment
    from pydub.playback import play
    import base64
    import io

    # Decode base64 to bytes
    audio_bytes = base64.b64decode(audio_content)

    # Create a file-like object from bytes
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio from the buffer
    audio = AudioSegment.from_file(audio_buffer, format=format)

    # Play the audio
    play(audio)


# Content types
class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """
    Internal representation of an image.
    Ideally normalized to base64 or a stable URL before reaching here.
    """

    type: Literal["image_url"] = "image_url"
    url: str  # data:image/png;base64,... or https://...
    detail: Literal["auto", "low", "high"] = "auto"


class AudioContent(BaseModel):
    """
    Internal representation of input audio.
    """

    type: Literal["input_audio"] = "input_audio"
    data: str  # Base64 encoded audio
    format: Literal["wav", "mp3"] = "mp3"


class AudioOutput(BaseModel):
    """Native audio response (e.g. GPT-4o-audio)."""

    id: str
    data: str  # Base64 encoded audio
    transcript: str | None = None  # The text representation
    format: Literal["wav", "mp3", "pcm16"] = "wav"


class ImageOutput(BaseModel):
    """Generated image response (e.g. DALL-E 3)."""

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None  # DALL-E often rewrites the prompt


# All possible user content types
Content = str | dict | list[TextContent | ImageContent | AudioContent | str]


# tool primitive
class ToolCall(BaseModel):
    """
    Represents a request from the Assistant to execute a function.
    """

    tool_call_id: uuid.UUID = Field(
        default_factory=lambda: uuid.UUID(int=uuid.uuid4().int)
    )
    type: Literal["function"] = "function"
    function_name: str
    arguments: dict[str, Any]


# message types
class Message(BaseModel):
    """
    Base class for all message types.
    Use this for isinstance checks.
    """

    role: Role
    content: Content | None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    message_id: uuid.UUID = Field(
        default_factory=lambda: uuid.UUID(int=uuid.uuid4().int)
    )

    @model_validator(mode="before")
    @classmethod
    def _disallow_base_instantiation(cls, data):
        if cls is Message:
            raise TypeError(
                "Message is an abstract base class and cannot be instantiated directly. Use SystemMessage, UserMessage, AssistantMessage, or ToolMessage. If you need to discriminate between them, use MessageUnion."
            )
        return data

    @property
    def time(self) -> str:
        """
        Human-readable time string.
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp / 1000))

    @override
    def __hash__(self) -> int:
        """
        Hash for caching and identity checks.
        """
        return hash(self.role.value + str(self.content))


class SystemMessage(Message):
    """
    System instructions.
    """

    role: Role = Role.SYSTEM
    role_str: Literal["system"] = "system"
    content: Content | None


class UserMessage(Message):
    """
    Messages sent by the human.
    Supports simple strings or complex multimodal chains (Text + Image).
    """

    role: Role = Role.USER
    role_str: Literal["user"] = "user"
    content: Content | None
    name: str | None = None


class AssistantMessage(Message):
    """
    Messages generated by the LLM.
    Supports Text, Reasoning, Tools, Audio, Images, and Structured Objects.
    """

    role: Role = Role.ASSISTANT
    role_str: Literal["assistant"] = "assistant"

    # text
    content: Content | None = None
    reasoning: str | None = None

    # action
    tool_calls: list[ToolCall] | None = None

    # multimodal
    audio: AudioOutput | None = None
    images: list[ImageOutput] | None = None

    # Special structured output (e.g. parsed JSON, XML, etc.) from Instructor
    parsed: BaseModel | list[BaseModel] | None = Field(default=None, exclude=True)

    @functools.cached_property
    def perplexity_content(self) -> Any:
        """
        Lazy factory for Perplexity responses with citations.
        If content has Perplexity structure (text + citations), construct rich PerplexityContent object.
        Import happens inside method for encapsulation and to avoid circular dependencies.

        Returns:
            PerplexityContent object if content matches Perplexity structure, None otherwise.
        """
        if not isinstance(self.content, dict):
            return None

        # Import inside method to avoid circular dependency
        from conduit.core.clients.perplexity.perplexity_content import (
            PerplexityContent,
            PerplexityCitation,
        )

        # Case 1: Text response with full structure
        if "text" in self.content and "citations" in self.content:
            citations = [
                PerplexityCitation(**c) for c in self.content["citations"]
            ]
            return PerplexityContent(
                text=self.content["text"],
                citations=citations,
            )

        # Case 2: Structured response - citations only
        if "citations" in self.content and self.parsed is not None:
            # Use JSON representation of structured object as "text"
            text = self.parsed.model_dump_json(indent=2)
            citations = [
                PerplexityCitation(**c) for c in self.content["citations"]
            ]
            return PerplexityContent(
                text=text,
                citations=citations,
            )

        return None

    @model_validator(mode="after")
    def validate_structure(self) -> AssistantMessage:
        # We need explicit presence of at least one payload
        has_payload = any(
            [
                self.content,
                self.tool_calls,
                self.reasoning,
                self.audio,
                self.images,
                self.parsed,
            ]
        )

        if not has_payload:
            raise ValueError(
                "AssistantMessage must have at least one of: content, tool_calls, reasoning, audio, images, or parsed output."
            )
        return self


class ToolMessage(Message):
    """
    The result of a tool execution, fed back to the LLM.
    """

    role: Role = Role.TOOL
    role_str: Literal["tool"] = "tool"
    content: str  # The output of the tool (usually JSON stringified)
    tool_call_id: str  # Links this result to the Assistant's ToolCall.id
    name: str | None = None  # Optional: name of the tool function


# discriminated union
MessageUnion = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role_str"),
]
