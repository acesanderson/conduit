from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.anthropic.payload import AnthropicPayload
from conduit.core.clients.anthropic.adapter import convert_message_to_anthropic
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage
from anthropic import AsyncAnthropic, AsyncStream
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
import os
import time

if TYPE_CHECKING:
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message
    from conduit.domain.result.result import GenerationResult


class AnthropicClient(Client):
    """
    Client implementation for Anthropic's Claude API.
    Async only.
    """

    def __init__(self):
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncAnthropic = raw_client

    @override
    def _initialize_client(self) -> tuple[Instructor, AsyncAnthropic]:
        """
        Creates both raw and instructor-wrapped clients.
        Raw client for standard completions, Instructor for structured responses.
        Uses instructor.from_anthropic() for Anthropic-specific handling.
        """
        raw_client = AsyncAnthropic(api_key=self._get_api_key())
        instructor_client = instructor.from_anthropic(raw_client)
        return instructor_client, raw_client

    @override
    def _get_api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("No ANTHROPIC_API_KEY found in environment variables")
        else:
            return api_key

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into Anthropic's specific dictionary format.
        """
        return convert_message_to_anthropic(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Anthropic's SDK.

        Key Anthropic-specific handling:
        - System messages are extracted from the messages list and passed as a separate 'system' parameter
        - max_tokens is required (defaults to 4096 if not specified)
        """
        # Convert messages and extract system messages
        converted_messages = []
        system_messages = []

        for message in request.messages:
            # Import here to avoid circular dependency
            from conduit.domain.message.message import SystemMessage

            if isinstance(message, SystemMessage):
                system_messages.append(message.content)
            else:
                converted_messages.append(self._convert_message(message))

        # Combine system messages into a single system parameter
        system_content = "\n\n".join(system_messages) if system_messages else None

        anthropic_payload = AnthropicPayload(
            model=request.params.model,
            messages=converted_messages,
            max_tokens=request.params.max_tokens if request.params.max_tokens else 4096,
            system=system_content,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            stream=request.params.stream,
        )
        return anthropic_payload

    @override
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Get token count per official Anthropic api endpoint.
        """
        # We need a raw client for this, not the instructor wrapper
        anthropic_client = Anthropic(api_key=self._get_api_key())

        # CASE 1: Raw String (Benchmarking)
        # We wrap it in a user message to satisfy the API, then subtract the overhead.
        if isinstance(payload, str):
            messages = [{"role": "user", "content": payload}]
            response = anthropic_client.messages.count_tokens(
                model=model,
                messages=messages,
            )
            # Subtract standard overhead (approx 3 tokens for a single-turn user message)
            # to return the "pure" string weight.
            return max(0, response.input_tokens - 3)

        # CASE 2: Message History (Context Window Check)
        if isinstance(payload, list):
            # Convert Message objects to Anthropic dictionaries
            messages_payload = []
            for m in payload:
                try:
                    # Filter out messages that Anthropic doesn't support (e.g. Audio)
                    messages_payload.append(m.to_anthropic())
                except NotImplementedError:
                    continue

            # If filtration resulted in empty list (e.g. only audio messages), return 0
            if not messages_payload:
                return 0

            response = anthropic_client.messages.count_tokens(
                model=model,
                messages=messages_payload,
            )
            return response.input_tokens

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
        Generate text using Anthropic's Claude API and return a GenerationResponse.

        Returns:
            - GenerationResponse object for successful non-streaming requests
            - AsyncStream object for streaming requests
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Use the raw client for standard completions
        result = await self._raw_client.messages.create(**payload_dict)

        # Handle streaming response
        if isinstance(result, AsyncStream):
            # For streaming, return the AsyncStream object directly
            return result

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = result.model
        input_tokens = result.usage.input_tokens  # Anthropic uses input_tokens not prompt_tokens
        output_tokens = result.usage.output_tokens  # Anthropic uses output_tokens not completion_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(result, "stop_reason"):
            match result.stop_reason:
                case "end_turn":
                    stop_reason = StopReason.STOP
                case "max_tokens":
                    stop_reason = StopReason.LENGTH
                case "stop_sequence":
                    stop_reason = StopReason.STOP

        # Extract the text content (Anthropic-specific path)
        content = result.content[0].text  # Different from OpenAI: content[0].text not choices[0].message.content
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
        Generate a structured response using Anthropic's function calling and return a GenerationResponse.

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
        ) = await self._client.messages.create_with_completion(
            response_model=request.params.response_model, **payload_dict
        )

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = completion.model
        input_tokens = completion.usage.input_tokens
        output_tokens = completion.usage.output_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(completion, "stop_reason"):
            match completion.stop_reason:
                case "end_turn":
                    stop_reason = StopReason.STOP
                case "max_tokens":
                    stop_reason = StopReason.LENGTH
                case "stop_sequence":
                    stop_reason = StopReason.STOP

        # Create AssistantMessage with parsed structured data
        # For Anthropic, content[0].text may be None for structured responses
        content_text = completion.content[0].text if completion.content else None
        assistant_message = AssistantMessage(
            content=content_text,
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
