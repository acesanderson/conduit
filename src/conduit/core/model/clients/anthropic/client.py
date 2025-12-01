from __future__ import annotations
from conduit.model.clients.client_base import Client
from conduit.odometer.usage import Usage
from anthropic import Anthropic, AsyncAnthropic, Stream
from pydantic import BaseModel
from typing import TYPE_CHECKING, override
from abc import ABC
import instructor
from instructor import Instructor
import os

if TYPE_CHECKING:
    from conduit.request.request import Request
    from conduit.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.message.message import Message


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
    ) -> tuple[str | BaseModel | SyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_anthropic(),
                )
            )
        else:
            result = self._client.chat.completions.create(**request.to_anthropic())
        # Handle streaming response
        if isinstance(result, Stream):
            usage = Usage(
                input_tokens=0,
                output_tokens=0,
            )
            return result, usage
        # Capture usage
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
            pass
        if isinstance(result, BaseModel):
            return result, usage
        else:
            raise ValueError(
                f"Unexpected result type from Anthropic API: {type(result)}"
            )


class AnthropicClientAsync(AnthropicClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_async_client = AsyncAnthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_async_client)

    async def query(
        self,
        request: Request,
    ) -> tuple[str | BaseModel | AsyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            (
                structured_response,
                result,
            ) = await self._client.chat.completions.create_with_completion(
                **request.to_anthropic(),
            )
        else:
            result = await self._client.chat.completions.create(
                **request.to_anthropic()
            )
        # Capture usage
        if isinstance(result, Stream):
            usage = Usage(
                input_tokens=0,
                output_tokens=0,
            )
            # Handle streaming response if needed
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
            pass
        if isinstance(result, BaseModel):
            return result, usage
        else:
            raise ValueError(
                f"Unexpected result type from Anthropic API: {type(result)}"
            )
