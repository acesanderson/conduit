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
import logging
from pathlib import Path

# Rich Imports for UI Rendering
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.syntax import Syntax
from rich.rule import Rule
from rich.box import ROUNDED, HEAVY

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_file(cls, file_path: str | Path) -> ImageContent:
        """
        Load image from a file and encode as base64 data URL.
        """
        import base64
        import mimetypes

        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "image/png"  # Default to PNG
        data_url = f"data:{mime_type};base64,{image_b64}"
        return cls(url=data_url)


class AudioContent(BaseModel):
    """
    Internal representation of input audio.
    """

    type: Literal["input_audio"] = "input_audio"
    data: str  # Base64 encoded audio
    format: Literal["wav", "mp3"] = "mp3"

    @classmethod
    def from_file(cls, file_path: str | Path) -> AudioContent:
        """
        Load audio from a file and encode as base64.
        """
        import base64

        file_path = Path(file_path)
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return cls(data=audio_b64, format=file_path.suffix.lstrip("."))


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

    def _extract_text_content(self) -> str:
        """Helper to safely extract string payload from complex Content types."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts = []
            for item in self.content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, TextContent):
                    parts.append(item.text)
                elif isinstance(item, ImageContent):
                    parts.append("[Image Content]")
                elif isinstance(item, AudioContent):
                    parts.append("[Audio Content]")
            return "\n".join(parts)
        return str(self.content)

    @override
    def __str__(self) -> str:
        """
        The "Pipe" View.
        Returns the pure text payload for piping/clipboard.
        Non-text content is replaced with simple placeholders like [Image Content].
        """
        return self._extract_text_content()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """
        The "Pretty" View.
        Renders the message as a Rich renderable (Panel, Rule, etc).
        Subclasses should override this for specific styling.
        """
        # Fallback for base class (shouldn't really happen)
        yield Text(str(self))


class SystemMessage(Message):
    """
    System instructions.
    """

    role: Role = Role.SYSTEM
    role_str: Literal["system"] = "system"
    content: Content | None

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # System messages are subtle context markers
        yield Rule(style="dim white")
        yield Text(
            f"⚙️ System Context • {self._extract_text_content()[:50]}...",
            style="dim italic",
            justify="center",
        )
        yield Text("")  # Spacing


class UserMessage(Message):
    """
    Messages sent by the human.
    Supports simple strings or complex multimodal chains (Text + Image).
    """

    role: Role = Role.USER
    role_str: Literal["user"] = "user"
    content: Content | None
    name: str | None = None

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text_content = self._extract_text_content()

        # Header info
        header = f"👤 User • [dim]{self.time}[/dim]"

        # Handle complex content display for UI (beyond just text extraction)
        renderables = []
        if isinstance(self.content, list):
            # Check for images/audio to display placeholders
            images = [i for i in self.content if isinstance(i, ImageContent)]
            audio = [a for a in self.content if isinstance(a, AudioContent)]

            if images:
                renderables.append(
                    Text(f"📷 {len(images)} Image(s) Attached", style="cyan italic")
                )
            if audio:
                renderables.append(
                    Text(f"🎤 {len(audio)} Audio Clip(s) Attached", style="cyan italic")
                )
            if images or audio:
                renderables.append(Rule(style="dim"))

        # Main text
        renderables.append(Markdown(text_content))

        from rich.console import Group

        yield Panel(
            Group(*renderables),
            title=header,
            title_align="left",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )


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
        """
        if not isinstance(self.content, dict):
            return None

        from conduit.core.clients.perplexity.perplexity_content import (
            PerplexityContent,
            PerplexityCitation,
        )

        # Case 1: Text response with full structure
        if "text" in self.content and "citations" in self.content:
            citations = [PerplexityCitation(**c) for c in self.content["citations"]]
            return PerplexityContent(
                text=self.content["text"],
                citations=citations,
            )

        # Case 2: Structured response - citations only
        if "citations" in self.content and self.parsed is not None:
            # Use JSON representation of structured object as "text"
            text = self.parsed.model_dump_json(indent=2)
            citations = [PerplexityCitation(**c) for c in self.content["citations"]]
            return PerplexityContent(
                text=text,
                citations=citations,
            )

        return None

    @model_validator(mode="after")
    def validate_structure(self) -> AssistantMessage:
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

    @override
    def __str__(self) -> str:
        # Override for Assistant because it might have tool calls but no content
        if self.content:
            return self._extract_text_content()
        if self.tool_calls:
            # If purely a tool call message, return a representation of the calls
            return "\n".join(
                [f"[ToolCall: {tc.function_name}]" for tc in self.tool_calls]
            )
        return ""

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        from rich.console import Group

        renderables = []

        # 1. Reasoning (DeepSeek style)
        if self.reasoning:
            renderables.append(
                Panel(
                    Markdown(self.reasoning),
                    title="🧠 Chain of Thought",
                    style="dim",
                    border_style="dim",
                )
            )

        # 2. Main Content
        if self.content:
            renderables.append(Markdown(self._extract_text_content()))

        # 3. Tool Calls
        if self.tool_calls:
            for tool in self.tool_calls:
                tool_code = f"{tool.function_name}({tool.arguments})"
                renderables.append(
                    Panel(
                        Syntax(tool_code, "python", theme="monokai", word_wrap=True),
                        title=f"🛠️ Call: {tool.function_name}",
                        border_style="magenta",
                    )
                )

        # 4. Multimodal Outputs
        if self.images:
            renderables.append(
                Text(f"🎨 Generated {len(self.images)} Image(s)", style="cyan")
            )
        if self.audio:
            transcript = f" ({self.audio.transcript})" if self.audio.transcript else ""
            renderables.append(Text(f"🔊 Generated Audio{transcript}", style="cyan"))

        # 5. Structured Parsed Output
        if self.parsed:
            json_str = (
                self.parsed.model_dump_json(indent=2)
                if isinstance(self.parsed, BaseModel)
                else str(self.parsed)
            )
            renderables.append(
                Panel(
                    Syntax(json_str, "json", theme="monokai", word_wrap=True),
                    title="🧩 Structured Output",
                    border_style="cyan",
                )
            )

        yield Panel(
            Group(*renderables),
            title=f"🤖 Assistant • [dim]{self.time}[/dim]",
            title_align="left",
            border_style="blue",
            box=HEAVY,
            padding=(1, 2),
        )


class ToolMessage(Message):
    """
    The result of a tool execution, fed back to the LLM.
    """

    role: Role = Role.TOOL
    role_str: Literal["tool"] = "tool"
    content: str  # The output of the tool (usually JSON stringified)
    tool_call_id: str  # Links this result to the Assistant's ToolCall.id
    name: str | None = None  # Optional: name of the tool function

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # Tool outputs are technical details, usually code/json
        yield Panel(
            Syntax(self.content, "json", theme="monokai", word_wrap=True),
            title=f"🛠️ Tool Output: {self.name or 'Unknown'} • [dim]{self.time}[/dim]",
            title_align="left",
            border_style="magenta",
            box=ROUNDED,
        )


# discriminated union
MessageUnion = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role_str"),
]
