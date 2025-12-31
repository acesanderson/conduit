"""
 NOTE: Typing is a mess in python. We have two types for Message:
- Message: the base class for all messages, used for isinstance checks.
- MessageUnion: a discriminated union of all message types, used for parsing/serialization.

* Enums don't work as discriminators in pydantic, so we use a new field 'role_str' for that purpose.
"""

from __future__ import annotations
import base64
import functools
import logging
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Literal, Any, Annotated, override, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator
from conduit.domain.message.role import Role

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

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


# tool primitive -- see conduit.capabilities.tool.Tool for more its mirror
type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]


class ToolCall(BaseModel):
    """
    Canonical request from the assistant to execute a function tool.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["function"] = "function"

    function_name: str
    arguments: dict[str, JsonValue]

    # Optional but very useful for adapters/debugging
    provider: str | None = None  # "openai" | "anthropic" | "gemini" | "ollama" ...
    raw: dict[str, JsonValue] | None = None


# message types
class Message(BaseModel):
    """
    Base class for all message types.
    Use this for isinstance checks.
    """

    role: Role
    content: Content | None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # For threading/conversation continuity
    predecessor_id: str | None = None
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @model_validator(mode="after")
    def _validate_session_continuity(self):
        """
        Validate that child messages explicitly specify their session context.

        Ensures that if a message references a predecessor (forming a child node in a
        conversation tree), the session_id must be explicitly provided rather than
        auto-generated. This prevents orphaned child messages that lack proper session
        continuity context.
        """
        # We only care if a predecessor exists (it's a child node)
        if (
            self.predecessor_id is not None
            and "session_id" not in self.model_fields_set
        ):
            raise ValueError(
                "Orphaned Child: If 'predecessor_id' is supplied, you must explicitly provide the existing 'session_id'."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _disallow_base_instantiation(cls, data):
        if cls is Message:
            raise TypeError(
                "Message is an abstract base class. Use SystemMessage, UserMessage, AssistantMessage, or ToolMessage."
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
        """
        return self._extract_text_content()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """
        The "Pretty" View.
        """
        from rich.text import Text

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
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.box import ROUNDED

        yield Panel(
            Markdown(self._extract_text_content()),
            title=f"System • [dim]{self.time}[/dim]",
            title_align="left",
            border_style="white",
            box=ROUNDED,
            padding=(0, 2),
        )


class UserMessage(Message):
    """
    Messages sent by the human.
    """

    role: Role = Role.USER
    role_str: Literal["user"] = "user"
    content: Content | None
    name: str | None = None

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.rule import Rule
        from rich.text import Text
        from rich.console import Group
        from rich.box import ROUNDED

        text_content = self._extract_text_content()
        header = f"User • [dim]{self.time}[/dim]"

        renderables = []
        if isinstance(self.content, list):
            images = [i for i in self.content if isinstance(i, ImageContent)]
            audio = [a for a in self.content if isinstance(a, AudioContent)]

            if images:
                renderables.append(
                    Text(f"{len(images)} Image(s) Attached", style="cyan italic")
                )
            if audio:
                renderables.append(
                    Text(f"{len(audio)} Audio Clip(s) Attached", style="cyan italic")
                )
            if images or audio:
                renderables.append(Rule(style="dim"))

        renderables.append(Markdown(text_content))

        yield Panel(
            Group(*renderables),
            title=header,
            title_align="left",
            border_style="green",
            box=ROUNDED,
            padding=(0, 2),
        )


class AssistantMessage(Message):
    """
    Messages generated by the LLM.
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

    # structured output
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

        if "text" in self.content and "citations" in self.content:
            citations = [PerplexityCitation(**c) for c in self.content["citations"]]
            return PerplexityContent(
                text=self.content["text"],
                citations=citations,
            )

        if "citations" in self.content and self.parsed is not None:
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
        if self.content:
            return self._extract_text_content()
        if self.tool_calls:
            return "\n".join(
                [f"[ToolCall: {tc.function_name}]" for tc in self.tool_calls]
            )
        return ""

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.syntax import Syntax
        from rich.text import Text
        from rich.console import Group
        from rich.box import HEAVY

        renderables = []

        # 1. Reasoning
        if self.reasoning:
            renderables.append(
                Panel(
                    Markdown(self.reasoning),
                    title="Chain of Thought",
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
                        title=f"Call: {tool.function_name}",
                        border_style="yellow",
                    )
                )

        # 4. Multimodal Outputs
        if self.images:
            renderables.append(
                Text(f"Generated {len(self.images)} Image(s)", style="cyan")
            )
        if self.audio:
            transcript = f" ({self.audio.transcript})" if self.audio.transcript else ""
            renderables.append(Text(f"Generated Audio{transcript}", style="cyan"))

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
                    title="Structured Output",
                    border_style="cyan",
                )
            )

        yield Panel(
            Group(*renderables),
            title=f"Assistant • [dim]{self.time}[/dim]",
            title_align="left",
            border_style="blue",
            box=HEAVY,
            padding=(0, 2),
        )


class ToolMessage(Message):
    """
    The result of a tool execution, fed back to the LLM.
    """

    role: Role = Role.TOOL
    role_str: Literal["tool"] = "tool"
    content: str  # The output of the tool (stringified)
    tool_call_id: str  # Links this result to the Assistant's ToolCall.id
    name: str | None = None  # Optional: name of the tool function

    @classmethod
    def from_result(
        cls, result: Any, tool_call_id: str, name: str | None = None
    ) -> ToolMessage:
        """
        Factory to create a ToolMessage from any raw Python result (dict, list, pydantic model).
        This handles the serialization so your tool functions don't have to.
        """
        import json

        if isinstance(result, (dict, list, int, float, bool, type(None))):
            # Standard JSON serialization
            content = json.dumps(result, default=str)
        elif isinstance(result, BaseModel):
            # Pydantic serialization
            content = result.model_dump_json()
        else:
            # Fallback for strings or unknown objects
            content = str(result)

        return cls(content=content, tool_call_id=tool_call_id, name=name)

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        import json
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.markdown import Markdown
        from rich.box import ROUNDED

        # Attempt to parse strictly for display purposes
        try:
            # Check if it's valid JSON
            json.loads(self.content)
            # It is JSON: use Syntax highlighting
            content_renderable = Syntax(
                self.content, "json", theme="monokai", word_wrap=True
            )
        except (ValueError, TypeError):
            # It is NOT JSON: treat as plain text/markdown
            content_renderable = Markdown(self.content)

        yield Panel(
            content_renderable,
            title=f"Tool Output: {self.name or 'Unknown'} • [dim]{self.time}[/dim]",
            title_align="left",
            border_style="yellow",
            box=ROUNDED,
        )


# discriminated union
MessageUnion = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role_str"),
]

if __name__ == "__main__":
    # Test rich rendering
    from rich.console import Console

    console = Console()

    messages: list[Message] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hello, can you show me a picture of a cat?"),
        AssistantMessage(
            content=[
                TextContent(text="Sure! Here is a picture of a cat:"),
            ]
        ),
        ToolMessage(
            content='{"result": "Tool executed successfully."}',
            tool_call_id=str(uuid.uuid4()),
            name="fetch_cat_image",
        ),
    ]

    for msg in messages:
        console.print(msg)
