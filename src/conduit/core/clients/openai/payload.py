from pydantic import BaseModel, Field, ConfigDict
from typing import Any


class OpenAIPayload(BaseModel):
    """
    Anti-corruption Layer for OpenAI chat completions.
    Validates top-level configuration while allowing flexibility in message structure.
    Since we use OpenAI spec for ollama, Gemini, and Perplexity, we will inherit from this model.
    """

    model_config = ConfigDict(extra="allow")  # Allow extra fields in future

    # Required
    model: str
    messages: list[dict[str, Any]]

    # Optional
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Tools and output
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
