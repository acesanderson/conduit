from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.storage.odometer.usage import Usage
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.anthropic.payload import AnthropicPayload
from conduit.core.clients.anthropic.adapter import convert_message_to_anthropic
from anthropic import (
    Anthropic,
    AsyncAnthropic,
    Stream,
    AsyncStream as AnthropicAsyncStream,
)
from typing import TYPE_CHECKING, override, Any
from abc import ABC
import instructor
from instructor import Instructor
import os

if TYPE_CHECKING:
    from conduit.domain.request.request import Request
    from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.domain.message.message import Message
    from conduit.domain.result.result import ConduitResult


class AnthropicClient(Client, ABC):
    def __init__(self):
        self._client: Instructor = self._initialize_client()

    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

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
    def _convert_request(self, request: Request) -> Payload:
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
            model=request.model,
            messages=converted_messages,
            max_tokens=request.max_tokens if request.max_tokens else 4096,
            system=system_content,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
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


class AnthropicClientSync(AnthropicClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

    @override
    def query(
        self,
        request: Request,
    ) -> ConduitResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Now, make the call
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Anthropic, so we use `create_with_completion`
            structured_response, result = self._client.messages.create_with_completion(
                response_model=request.response_model, **payload_dict
            )
        else:
            # Use the standard completion method
            result = self._client.messages.create(
                response_model=request.response_model, **payload_dict
            )

        # Capture usage
        if isinstance(result, Stream):
            # Handle streaming response if needed; usage is handled by StreamParser
            usage = Usage(input_tokens=0, output_tokens=0)
            return result, usage

        usage = Usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )

        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage

        # First try to get text content from the result
        try:
            result = result.content[0].text
            return result, usage
        except AttributeError:
            # If the result is a BaseModel or Stream, handle accordingly
            pass

        return result, usage


class AnthropicClientAsync(AnthropicClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_async_client = AsyncAnthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_async_client)

    @override
    async def query(
        self,
        request: Request,
    ) -> ConduitResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Now, make the call
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Anthropic, so we use `create_with_completion`
            (
                structured_response,
                result,
            ) = await self._client.messages.create_with_completion(
                response_model=request.response_model, **payload_dict
            )
        else:
            # Use the standard completion method
            result = await self._client.messages.create(
                response_model=request.response_model, **payload_dict
            )

        # Capture usage
        if isinstance(result, AnthropicAsyncStream):
            # Handle streaming response if needed; usage is handled by StreamParser
            usage = Usage(input_tokens=0, output_tokens=0)
            return result, usage

        usage = Usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )

        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage

        # First try to get text content from the result
        try:
            result = result.content[0].text
            return result, usage
        except AttributeError:
            # If the result is a BaseModel or AsyncStream, handle accordingly
            pass

        return result, usage
