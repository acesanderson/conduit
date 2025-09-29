from pydantic import BaseModel, Field
from conduit.model.models.provider import Provider
from typing import Optional, ClassVar


class ClientParams(BaseModel):
    """
    Parameters that are specific to a client.
    """

    provider: ClassVar[Provider]


class OpenAIParams(ClientParams):
    """
    Parameters specific to OpenAI API spec-using clients.
    NOTE: This is a generic OpenAI client, not the official OpenAI API.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)
    provider: ClassVar[Provider] = "openai"

    # Core parameters
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None


class GoogleParams(OpenAIParams):
    """
    Parameters specific to Gemini clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)
    provider: ClassVar[Provider] = "google"


class OllamaParams(OpenAIParams):
    """
    Parameters specific to Ollama clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    Ollama has a special implementation of client_params, see the special handling within Params.to_openai_spec().
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)
    provider: ClassVar[Provider] = "ollama"

    # Core parameters -- note for Instructor we will need to embed these in "extra_body":{"options": {}}
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[list[str]] = None


class AnthropicParams(ClientParams):
    """
    Parameters specific to Anthropic clients.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)
    provider: ClassVar[Provider] = "anthropic"

    # Core parameters
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None

    # Excluded from serialization in API calls
    model: str = Field(
        default="",
        description="The model identifier to use for inference.",
        exclude=True,
    )


class PerplexityParams(OpenAIParams):
    """
    Parameters specific to Perplexity clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)
    provider: ClassVar[Provider] = "perplexity"


# All client parameters types, for validation
ClientParamsModels = [
    OpenAIParams,
    OllamaParams,
    AnthropicParams,
    GoogleParams,
    PerplexityParams,
]
