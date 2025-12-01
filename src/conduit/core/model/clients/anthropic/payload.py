from pydantic import BaseModel, Field, ConfigDict
from typing import Any


class AnthropicPayload(BaseModel):
    """
    Anti-corruption Layer for Anthropic Messages API.
    Validates top-level configuration while allowing flexibility in message structure.

    Key differences from OpenAI:
    - 'max_tokens' is REQUIRED (we default to 4096 if not set).
    - 'system' is a top-level parameter, not a message role.
    - 'stop_sequences' instead of 'stop'.
    - 'thinking' parameter for Claude 3.7+ reasoning.
    """

    model_config = ConfigDict(extra="allow")

    # Required
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = Field(
        default=4096, ge=1
    )  # Anthropic throws an error if max_tokens is missing.

    system: str | list[dict[str, Any]] | None = None  # Quirk of Anthropic API

    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)

    # Anthropic uses 'stop_sequences' (list), not 'stop'
    stop_sequences: list[str] | None = None
    stream: bool | None = None

    # Advanced features
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    # Tools
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
