"""
For Google Gemini models.
"""

from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.google.payload import GooglePayload
from conduit.core.clients.google.adapter import convert_message_to_google
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage
from openai import AsyncOpenAI, AsyncStream
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
from abc import ABC
import os
import time

if TYPE_CHECKING:
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message


class GoogleClient(Client, ABC):
    """
    Client implementation for Google's Gemini API using the OpenAI-compatible endpoint.
    Async by default.
    """

    def __init__(self):
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client

    @override
    def _initialize_client(self) -> tuple[Instructor, AsyncOpenAI]:
        """
        Creates both raw and instructor-wrapped clients.
        Raw client for standard completions, Instructor for structured responses.
        """
        raw_client = AsyncOpenAI(
            api_key=self._get_api_key(),
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )
        instructor_client = instructor.from_openai(
            raw_client,
            mode=instructor.Mode.JSON,
        )
        return instructor_client, raw_client

    @override
    def _get_api_key(self) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return api_key

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into Google's specific dictionary format.
        Since Google uses OpenAI spec, we delegate to the OpenAI adapter.
        """
        return convert_message_to_google(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Google's SDK (via OpenAI spec).
        """
        # Load client params
        client_params = request.params.client_params or {}
        allowed_params = {"frequency_penalty", "presence_penalty"}
        for param in client_params.keys():
            if param not in allowed_params:
                raise ValueError(f"Unsupported Google client parameter: {param}")
        # Convert messages
        converted_messages = self._convert_messages(request.messages)
        # Build payload
        google_payload = GooglePayload(
            model=request.params.model,
            messages=converted_messages,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            # Google-specific params
            **client_params,
        )
        return google_payload

    @override
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Get the token count per official tokenizer (through Google Native API).
        We use the google.generativeai SDK for this because Gemini tokens != tiktoken.
        """
        import google.generativeai as genai

        # Configure the native SDK with the API key
        genai.configure(api_key=self._get_api_key())
        model_client = genai.GenerativeModel(model_name=model)

        # CASE 1: Raw String
        if isinstance(payload, str):
            response = model_client.count_tokens(payload)
            return response.total_tokens

        # CASE 2: Message History
        if isinstance(payload, list):
            # We must convert conduit Messages (OpenAI-style) to Google Native format
            # just for the token counter.
            # Google Native Format: [{'role': 'user'|'model', 'parts': ['...']}]
            native_contents = []

            for msg in payload:
                # Map roles: 'assistant' -> 'model', everything else -> 'user'
                role = "model" if msg.role == "assistant" else "user"

                parts = []
                # Extract text content safely
                if hasattr(msg, "text_content") and msg.text_content:
                    parts.append(msg.text_content)
                elif isinstance(msg.content, str):
                    parts.append(msg.content)

                # Note: If you want to count image tokens accurately, you would need to
                # convert the base64 to a PIL image or Blob and append it to parts here.
                # For now, we are counting the text weight of the conversation.

                if parts:
                    native_contents.append({"role": role, "parts": parts})

            if not native_contents:
                return 0

            response = model_client.count_tokens(native_contents)
            return response.total_tokens

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
        Generate text using Google's Gemini API and return a GenerationResponse.

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
        Generate a structured response using Google's function calling and return a GenerationResponse.

        Returns:
            - GenerationResponse object with parsed structured data in AssistantMessage.parsed
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Make the API call with function calling
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
