"""
Client subclass for Ollama models.
This doesn't require an API key since these are locally hosted models.
We can use openai api calls to the ollama server, but we use the instructor library to handle the API calls.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
We define preferred defaults for context sizes in a separate json file.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.ollama.payload import OllamaPayload
from conduit.core.clients.ollama.adapter import convert_message_to_ollama
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage
from openai import AsyncOpenAI, AsyncStream
from collections import defaultdict
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
import json
import logging
import time

if TYPE_CHECKING:
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class OllamaClient(Client):
    """
    Client implementation for Ollama models using OpenAI-compatible endpoint.
    Async only.
    """

    def __init__(self):
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client
        self.update_ollama_models()  # This allows us to keep the model file up to date.

    # Load Ollama context sizes from the JSON file
    try:
        settings.paths["OLLAMA_CONTEXT_SIZES_PATH"].parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(settings.paths["OLLAMA_CONTEXT_SIZES_PATH"]) as f:
            _ollama_context_data = json.load(f)

        # Use defaultdict to set default context size to 4096 if not specified
        _ollama_context_sizes = defaultdict(lambda: 32768)
        _ollama_context_sizes.update(_ollama_context_data)
    except Exception:
        logger.warning(
            f"Could not load Ollama context sizes from {settings.paths['OLLAMA_CONTEXT_SIZES_PATH']}. Using default of 32768."
        )
        _ollama_context_sizes = defaultdict(lambda: 32768)

    @override
    def _initialize_client(self) -> tuple[Instructor, AsyncOpenAI]:
        """
        Creates both raw and instructor-wrapped clients.
        Raw client for standard completions, Instructor for structured responses.
        """
        raw_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required but unused
        )
        instructor_client = instructor.from_openai(
            raw_client,
            mode=instructor.Mode.JSON,
        )
        return instructor_client, raw_client

    @override
    def _get_api_key(self) -> str:
        """
        Best thing about Ollama; no API key needed.
        """
        return ""

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into Ollama's specific dictionary format.
        Since Ollama uses OpenAI spec, we delegate to the OpenAI adapter.
        """
        return convert_message_to_ollama(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Ollama's SDK (via OpenAI spec).
        Ollama supports extra_body for additional configuration options.
        """
        # load client_params
        client_params = request.params.client_params or {}
        allowed_params = {"num_ctx"}
        for param in client_params:
            if param not in allowed_params:
                raise ValueError(f"Ollama does not support client param: {param}")
        # convert messages
        converted_messages = self._convert_messages(request.messages)
        # build payload
        ollama_payload = OllamaPayload(
            model=request.params.model,
            messages=converted_messages,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            # ollama specific params (nested under extra_body)
            extra_body={**client_params} if client_params else None,
        )
        return ollama_payload

    def update_ollama_models(self):
        """
        Updates the list of Ollama models.
        We run is every time ollama is initialized.
        """
        import ollama

        # Ensure the directory exists
        settings.paths["OLLAMA_MODELS_PATH"].parent.mkdir(parents=True, exist_ok=True)
        # Lazy load ollama module
        ollama_models = [m["model"] for m in ollama.list()["models"]]
        ollama_model_dict = {"ollama": ollama_models}
        with open(settings.paths["OLLAMA_MODELS_PATH"], "w") as f:
            json.dump(ollama_model_dict, f)

    @override
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Count tokens using Ollama's API via the official library.
        We set "num_predict" to 0 so we only process the prompt/history and get the eval count.
        """
        import ollama

        # CASE 1: Raw String
        if isinstance(payload, str):
            response = ollama.generate(
                model=model,
                prompt=payload,
                options={"num_predict": 0},  # Minimal generation
            )
            return int(response.get("prompt_eval_count", 0))

        # CASE 2: Message History
        if isinstance(payload, list):
            # Convert internal Messages to OpenAI/Ollama compatible dicts
            messages_payload = [m.to_ollama() for m in payload]

            response = ollama.chat(
                model=model,
                messages=messages_payload,
                options={"num_predict": 0},
            )
            return int(response.get("prompt_eval_count", 0))

        raise ValueError("Payload must be string or list[Message]")

    @override
    async def query(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        match request.params.output_type:
            case "text":
                return await self._generate_text(request)
            case "structured_response":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(
                    f"Unsupported output type: {request.params.output_type}"
                )

    async def _generate_text(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate text using Ollama and return a GenerationResponse.

        Returns:
            - GenerationResponse object for successful non-streaming requests
            - AsyncStream object for streaming requests
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Use the raw client for standard completions
        result = await self._raw_client.chat.completions.create(**payload_dict)

        # Handle streaming response
        if isinstance(result, AsyncStream):
            # For streaming, return the AsyncStream object directly
            return result

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = result.model
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(result.choices[0], "finish_reason"):
            finish_reason = result.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "content_filter":
                stop_reason = StopReason.CONTENT_FILTER

        # Extract the text content
        content = result.choices[0].message.content
        assistant_message = AssistantMessage(content=content)

        # Create ResponseMetadata
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_stem,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate a structured response using Ollama's function calling and return a GenerationResponse.

        Returns:
            - GenerationResponse object with parsed structured data in AssistantMessage.parsed
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Make the API call with function calling using instructor client
        (
            user_obj,
            completion,
        ) = await self._client.chat.completions.create_with_completion(
            response_model=request.params.response_model, **payload_dict
        )

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = completion.model
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(completion.choices[0], "finish_reason"):
            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH

        # Create AssistantMessage with parsed structured data
        assistant_message = AssistantMessage(
            content=completion.choices[0].message.content,
            parsed=user_obj,
        )

        # Create ResponseMetadata
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_stem,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )
