from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from Chain.message.message import Message
from Chain.message.textmessage import TextMessage
from Chain.message.messages import Messages
from Chain.parser.parser import Parser
from Chain.progress.verbosity import Verbosity
from Chain.progress.display_mixins import (
    RichDisplayParamsMixin,
    PlainDisplayParamsMixin,
)
from Chain.request.clientparams import (
    ClientParamsModels,
    OpenAIParams,
    OllamaParams,
    AnthropicParams,
    GoogleParams,
    PerplexityParams,
)
from Chain.request.outputtype import OutputType
from Chain.model.models.provider import Provider


class Request(BaseModel, RichDisplayParamsMixin, PlainDisplayParamsMixin):
    """
    Parameters that are constructed by Model and are sent to Clients.
    Note: we mixin our DisplayParamsMixin classes to provide rich and plain display methods.
    """

    # Core parameters
    output_type: OutputType = Field(
        default="text", description="Desired output: 'text', 'image', 'audio'"
    )
    model: str = Field(..., description="The model identifier to use for inference.")
    messages: Messages | list[Message] = Field(
        default_factory=list,
        description="List of messages to send to the model. Can include text, images, audio, etc.",
    )

    # Optional parameters
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for sampling. If None, defaults to provider-specific value.",
    )
    stream: bool = False
    verbose: Verbosity = Field(
        default=Verbosity.PROGRESS,
        exclude=True,
        description="Verbosity level for logging and progress display.",
    )
    response_model: type[BaseModel] | dict | None = Field(
        default=None,
        description="Pydantic model to convert messages to a specific format. If dict, this is a schema for the model for serialization / caching purposes.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate. If None, it is not passed to the client (except for Anthropic, which requires it and has a default).",
    )
    # Post model init parameters
    provider: Optional[Provider] = Field(
        default=None,
        description="Provider of the model, populated post init. Not intended for direct use.",
    )
    # Client parameters (embedded in dict for now)
    client_params: Optional[dict] = Field(
        default=None,
        description="Client-specific parameters. Validated against the provider-approved params, through our ClientParams.",
    )

    def model_post_init(self, __context) -> None:
        """
        Post-initialization method to validate and set parameters.
        Validates:
        - Model is supported by at least one provider
        - Temperature is within range for the provider
        - Client parameters match the provider
        Sets:
        - Provider based on the model
        - Client parameters based on the provider
        - Messages are converted to a consistent format
        Finally, validates that we have our required parameters.
        """
        self._validate_model()  # Raise error if model is not supported
        self._set_provider()  # Set provider based on model
        self._validate_temperature()  # Raise error if temperature is out of range for provider
        self._set_client_params()  # Set client_params based on provider
        # Convert messages to Messages object if they are a list
        self.messages = (
            Messages(self.messages)
            if isinstance(self.messages, list)
            else self.messages
        )
        # Validate the whole lot
        if self.messages is None or len(self.messages) == 0:
            raise ValueError("Messages cannot be empty. Likely a code error.")
        if self.provider == "" or self.provider is None:
            raise ValueError(
                "Provider must be set based on the model. Likely a code error."
            )

    # Constructor methods
    @classmethod
    def from_query_input(
        cls, query_input: str | Message | list[Message], **kwargs
    ) -> "Request":
        """
        Create a Request from various input types.

        Args:
            query_input: Can be:
                - str: Creates a TextMessage with this content
                - Message: Uses the Message directly (AudioMessage, ImageMessage, etc.)
                - list[Message]: Uses the list of messages directly
        """
        messages = kwargs.pop("messages", [])

        # Handle different input types
        if isinstance(query_input, str):
            # Original behavior - create TextMessage from string
            user_message = TextMessage(role="user", content=query_input)
            modified_messages = messages + [user_message]

        elif isinstance(query_input, Message):
            # ✅ NEW: Handle Message objects directly (AudioMessage, ImageMessage, etc.)
            modified_messages = messages + [query_input]

        elif isinstance(query_input, list) and all(
            isinstance(msg, Message) for msg in query_input
        ):
            # ✅ NEW: Handle list of Message objects
            modified_messages = messages + query_input

        else:
            raise ValueError(
                f"query_input must be str, Message, or list[Message], got {type(query_input)}"
            )

        kwargs.update({"messages": modified_messages})
        return cls(**kwargs)

    # Validation methods
    def _set_provider(self):
        from Chain.model.models.modelstore import ModelStore

        for provider in ModelStore.models().keys():
            if self.model in ModelStore.models()[provider]:
                self.provider = provider
                return
        if not self.provider:
            raise ValueError(f"Model '{self.model}' is not supported by any provider.")

    def _set_client_params(self):
        """
        User should add extra client params as a dict called client_params.
        """
        # If client_params is already set, we assume it's a dict and validate it.
        if self.client_params is None:
            return
        # If client_params is a dict, we need to validate it against the provider's client params.
        for client_param_type in ClientParamsModels:
            if self.provider == client_param_type.provider:
                # Validate the dict against the provider's client params
                try:
                    client_param_type.model_validate(self.client_params)
                except ValidationError as e:
                    raise ValidationError(
                        f"Client parameters do not match the expected format for provider '{self.provider}': {e}"
                    )

    def _validate_temperature(self):
        """
        Validate temperature against provider-specific ranges.
        """
        if self.temperature is None:
            return
        # Find the temperature range for the provider
        temperature_range: Optional[tuple[float, float]] = None
        for client_param_type in ClientParamsModels:
            if self.provider == client_param_type.provider:
                temperature_range = client_param_type.temperature_range
        if not temperature_range:
            raise ValueError(
                f"Temperature range not found for provider: {self.provider}"
            )
        # Check if the temperature is within the range
        if temperature_range[0] <= self.temperature <= temperature_range[1]:
            return
        else:
            raise ValidationError(
                f"Temperature {self.temperature} is out of range {temperature_range} for provider: {self.provider}"
            )

    def _validate_model(self):
        """
        Validate that the model is supported by at least one provider.
        """
        from Chain.model.models.modelstore import ModelStore

        if not ModelStore.is_supported(self.model):
            raise ValidationError(f"Model '{self.model}' is not supported.")
        return

    # For Caching
    def normalize_json(self, obj):
        """
        Nested dicts and lists are normalized to ensure consistent ordering.
        This is crucial for consistent caching and comparison.
        """
        if isinstance(obj, dict):
            return {k: self.normalize_json(obj[k]) for k in sorted(obj)}
        elif isinstance(obj, list):
            return [self.normalize_json(item) for item in obj]
        else:
            return obj

    def generate_cache_key(self) -> str:
        """
        Generate a reliable cache key for the Request instance.
        Only includes fields that would affect the LLM response.
        """
        from hashlib import sha256
        import json

        # Use sort_keys for deterministic JSON ordering
        messages_str: str = json.dumps(self.messages.to_cache_dict(), sort_keys=True)
        # Include parser since it affects response format
        schema_str: str = (
            Parser.as_string(self.response_model) if self.response_model else "none"
        )
        # Handle None temperature gracefully
        temp_str = str(self.temperature) if self.temperature is not None else "none"
        # Include client_params if they are set; the normalize_json ensures consistent ordering, recursively
        if self.client_params:
            client_params_str = json.dumps(
                self.normalize_json(self.client_params), sort_keys=True
            )
        else:
            client_params_str = "none"
        params_str = "|".join(
            [
                self.model,
                messages_str,
                temp_str,
                schema_str,
                self.provider or "none",
                client_params_str,
            ]
        )
        return sha256(params_str.encode("utf-8")).hexdigest()

    # Dunder methods for string representation
    def __str__(self) -> str:
        """
        Generate a string representation of the Request instance.
        """
        return f"Request(model={self.model}, messages={self.messages}, temperature={self.temperature}, provider={self.provider})"

    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the Request instance.
        """
        return (
            f"Request(model={self.model!r}, messages={self.messages!r}, "
            f"temperature={self.temperature!r}, provider={self.provider!r}, "
            f"client_params={self.client_params!r}, response_model={self.response_model!r})"
        )

    # Serialization methods (for Cache, MessageStore, etc.)
    def to_cache_dict(self) -> dict:
        """
        Required attributes:
        - model: str
        - messages: list[dict]
        - temperature: Optional[float]
        - response_model: Optional[dict]
        - provider: Optional[str]
        - client_params: Optional[dict]
        - stream: bool
        - verbose: Verbosity
        """
        cache_dict = {
            "model": self.model,
            "messages": self.messages.to_cache_dict(),
            "temperature": self.temperature,
            "response_model": (
                Parser.as_string(self.response_model)
                if self.response_model and isinstance(self.response_model, type)
                else None
            ),
            "provider": self.provider,
            "client_params": self.client_params,
            "stream": self.stream,
            "verbose": self.verbose.value,
        }
        return cache_dict

    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "Request":
        model = cache_dict.get("model")
        messages = cache_dict.get("messages")
        temperature = cache_dict.get("temperature", None)
        response_model = cache_dict.get("response_model", None)
        if response_model and isinstance(response_model, str):
            # Find the matching Pydantic class in Parser._response_models
            import json

            try:
                schema_dict = json.loads(response_model)
                title = schema_dict.get("title")

                for model_class in Parser._response_models:
                    if model_class.__name__ == title:
                        response_model = model_class
                else:
                    # If not found, set to None to avoid validation error
                    response_model = None
            except (json.JSONDecodeError, KeyError):
                response_model = None

        client_params = cache_dict.get("client_params", None)
        stream = cache_dict.get("stream", False)
        verbose = cache_dict.get("verbose", Verbosity.PROGRESS)

        return cls(
            model=str(model),
            messages=Messages.from_cache_dict(messages) if messages else [],
            temperature=temperature,
            response_model=response_model,
            client_params=client_params,
            stream=stream,
            verbose=verbose,
        )

    # Each customization method first operates on client_params, then constructs base parameters.
    def _to_openai_spec(self) -> dict:
        """
        We use OpenAI spec with OpenAI, Gemini, Ollama, and Perplexity clients.
        """
        if self.client_params:
            assert OpenAIParams.model_validate(self.client_params), (
                f"OpenAIParams expected for OpenAI client, not {type(self.client_params)}."
            )

        # Convert messages to OpenAI format based on the provider
        match self.provider:
            case "openai":
                converted_messages = self.messages.to_openai()  # type: ignore
            case "ollama":
                converted_messages = self.messages.to_ollama()  # type: ignore
            case "google":
                converted_messages = self.messages.to_google()  # type: ignore
            case "perplexity":
                converted_messages = self.messages.to_perplexity()  # type: ignore
            case _:
                raise ValueError(
                    f"Unsupported provider '{self.provider}' for OpenAI spec conversion."
                )
        assert len(converted_messages) > 0, (
            "converted_messages is empty. Unable to convert to OpenAI spec."
        )

        base_params = {
            "model": self.model,
            "messages": converted_messages,
            "response_model": self.response_model,
            "temperature": self.temperature,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
        }

        # OpenAI has different max token parameters for different models, irritatingly.
        if self.max_tokens is not None:
            if self.provider == "openai" and (
                self.model.startswith("o1-")
                or self.model.startswith("o3-")
                or self.model.startswith("gpt-5")
            ):
                base_params["max_completion_tokens"] = self.max_tokens
                base_params.pop("max_tokens", None)  # Remove max_tokens if set
            else:
                # All other providers (google, ollama, perplexity) and
                # older OpenAI models use max_tokens
                base_params["max_tokens"] = self.max_tokens

        # Automatically include all client params
        if self.provider == "ollama":
            if self.client_params:
                # Ollama expects options to be nested under "extra_body"
                base_params.update({"extra_body": {"options": self.client_params}})
        else:
            if self.client_params:
                base_params.update(self.client_params)

        # Filter out None values and return
        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }  # Actually filter None values EXCEPT for response_model, as Instructor expects it to be present

    def to_openai(self) -> dict:
        if self.client_params:
            assert OpenAIParams.model_validate(self.client_params), (
                f"OpenAIParams expected for OpenAI client, not {type(self.client_params)}."
            )
        return self._to_openai_spec()

    def to_ollama(self) -> dict:
        """
        We should set num_ctx so we have maximal context window.
        Recall that Ollama with Instructor/OpenAI spec expects options to be nested under extra_body, so we have a special case within to_openai_spec.
        """
        from Chain.model.models.modelstore import ModelStore

        if self.client_params:
            assert OllamaParams.model_validate(self.client_params), (
                f"OllamaParams expected for Ollama client, not {type(self.client_params)}."
            )
        # Set num_ctx to the maximum context window for the model.
        num_ctx = ModelStore.get_num_ctx(self.model)
        if num_ctx is None:
            raise ValueError(
                f"Model '{self.model}' does not have a defined context window."
            )
        if self.client_params:
            self.client_params["num_ctx"] = num_ctx
        else:
            self.client_params = {"num_ctx": num_ctx}
        return self._to_openai_spec()

    def to_anthropic(self) -> dict:
        """
        Convert parameters to Anthropic format.
        Key differences from OpenAI:
        1. System messages become a separate 'system' parameter
        2. max_tokens is required
        3. No response_model in the API call params
        """
        if self.client_params:
            assert AnthropicParams.model_validate(self.client_params), (
                f"AnthropicParams expected for Anthropic client, not {type(self.client_params)}."
            )
        # Start with converted messages
        converted_messages = self.messages.to_anthropic()  # type: ignore

        # Extract system message if present
        system_content = ""
        filtered_messages = []

        for message in converted_messages:
            if message.get("role") == "system":
                system_content = message.get("content", "")
            else:
                filtered_messages.append(message)

        # Also check if any remaining messages have system role (Anthropic quirk)
        for message in filtered_messages:
            if message.get("role") == "system":
                # Convert system role to user role (as per your AnthropicClient logic)
                message["role"] = "user"

        # Build base parameters
        base_params = {
            "model": self.model,
            "messages": filtered_messages,
            "max_retries": 0,
            "response_model": self.response_model,
            "temperature": (
                self.temperature if self.temperature is not None else 1.0
            ),  # Default to 1.0 if not set
            "max_tokens": self.max_tokens if self.max_tokens is not None else 4000,
        }

        # Add system parameter if we have system content
        if system_content:
            base_params["system"] = system_content

        # Remove message_type as it's not used in Anthropic API
        for message in filtered_messages:
            if message.get("message_type"):
                message.pop("message_type")

        # Add temperature if specified and validate range
        if self.temperature is not None:
            if not (0 <= self.temperature <= 1):
                raise ValueError(
                    "Temperature for Anthropic models needs to be between 0 and 1."
                )
            base_params["temperature"] = self.temperature

        # Add client_params to base_params
        if self.client_params:
            base_params.update(self.client_params)

        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }

    def to_google(self) -> dict:
        if self.client_params:
            assert GoogleParams.model_validate(self.client_params), (
                f"GoogleParams expected for Google client, not {type(self.client_params)}."
            )
            # Remove "frequency_penalty" key if it exists, as Google does not use it (unlike OpenAI).
            self.client_params.pop("frequency_penalty", None)
        return self._to_openai_spec()

    def to_perplexity(self) -> dict:
        if self.client_params:
            assert PerplexityParams.model_validate(self.client_params), (
                f"PerplexityParams expected for Perplexity client, not {type(self.client_params)}."
            )
            # Remove "stop" key if it exists, as Perplexity does not use it (unlike OpenAI).
            self.client_params.pop("stop", None)
        return self._to_openai_spec()
