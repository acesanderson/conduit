"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
You want both the 'content' and 'citations' fields from the response object.
"""

from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.storage.odometer.usage import Usage
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.perplexity.payload import PerplexityPayload
from conduit.core.clients.perplexity.adapter import convert_message_to_perplexity
from conduit.core.clients.perplexity.perplexity_content import (
    PerplexityContent,
    PerplexityCitation,
)
from openai import OpenAI, AsyncOpenAI, Stream, AsyncStream as OpenAIAsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from abc import ABC
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
import json
import os

if TYPE_CHECKING:
    from conduit.domain.result.result import ConduitResult
    from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.domain.request.request import Request
    from conduit.domain.message.message import Message


class PerplexityClient(Client, ABC):
    """
    This is a base class; we have two subclasses: PerplexityClientSync and PerplexityClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client: Instructor = self._initialize_client()

    @override
    def _initialize_client(self) -> Instructor:
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    @override
    def _get_api_key(self) -> str:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set.")
        else:
            return api_key

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into Perplexity's specific dictionary format.
        Since Perplexity uses OpenAI spec, we delegate to the Perplexity adapter (which uses OpenAI).
        """
        return convert_message_to_perplexity(message)

    @override
    def _convert_request(self, request: Request) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Perplexity's SDK (OpenAI-compatible).

        Perplexity-specific parameters:
        - return_citations: Enable citation return
        - return_images: Enable image return
        - search_recency_filter: Time-based filtering
        """
        # load client_params
        client_params = request.client_params or {}
        allowed_params = {"return_citations", "search_recency_filter"}
        for param in client_params.keys():
            if param not in allowed_params:
                raise ValueError(
                    f"Invalid client_param '{param}' for PerplexityClient. Allowed params: {allowed_params}"
                )
        # convert messages
        converted_messages = self._convert_messages(request.messages)
        # build payload
        perplexity_payload = PerplexityPayload(
            model=request.model,
            messages=converted_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream,
            # Perplexity-specific params
            **client_params,
        )
        return perplexity_payload

    @override
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Return the token count for a string, per model's tokenization function.
        cl100k_base is good enough for Perplexity approximation.
        """
        import tiktoken

        # Perplexity models are often Llama-based, but cl100k_base is the
        # standard fallback for OpenAI-compatible APIs in Python without
        # heavy `transformers` dependencies.
        encoding = tiktoken.get_encoding("cl100k_base")

        # CASE 1: Raw String
        if isinstance(payload, str):
            return len(encoding.encode(payload))

        # CASE 2: Message History
        if isinstance(payload, list):
            # Lazy import to avoid circular dependency
            from conduit.domain.message.textmessage import TextMessage

            # Standard OpenAI-compatible overhead approximation
            tokens_per_message = 3
            num_tokens = 0

            for message in payload:
                num_tokens += tokens_per_message

                # Role
                num_tokens += len(encoding.encode(message.role))

                # Content
                # Perplexity generally only handles Text.
                # If ImageMessage/AudioMessage are passed, we only count their text content.
                content_str = ""
                if hasattr(message, "text_content") and message.text_content:
                    content_str = message.text_content
                elif isinstance(message.content, str):
                    content_str = message.content
                elif isinstance(message.content, list):
                    # Handle complex content (list of BaseModels) by dumping to string
                    try:
                        content_str = json.dumps(
                            [m.model_dump() for m in message.content]
                        )
                    except AttributeError:
                        content_str = str(message.content)
                elif isinstance(message.content, BaseModel):
                    try:
                        content_str = message.content.model_dump_json()
                    except AttributeError:
                        content_str = str(message.content)

                num_tokens += len(encoding.encode(content_str))

            num_tokens += 3  # reply primer
            return num_tokens
        raise ValueError("Payload must be string or list[Message]")


class PerplexityClientSync(PerplexityClient):
    @override
    def __init__(self):
        self._client: Instructor = self._initialize_client()

    @override
    def _initialize_client(self) -> Instructor:
        # Keep both raw and instructor clients
        self._raw_client = OpenAI(
            api_key=self._get_api_key(), base_url="https://api.perplexity.ai"
        )
        return instructor.from_perplexity(self._raw_client)

    @override
    def query(self, request: Request) -> ConduitResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Now, make the call
        structured_response = None
        if request.response_model is not None:
            # Use instructor for structured responses
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    response_model=request.response_model, **payload_dict
                )
            )
        else:
            # Use raw client for unstructured responses to access citations
            result = self._raw_client.chat.completions.create(**payload_dict)

        # Capture usage
        if isinstance(result, Stream):
            # Handle streaming response if needed; usage is handled by StreamParser
            usage = Usage(input_tokens=0, output_tokens=0)
            return result, usage

        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )

        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage

        if isinstance(result, ChatCompletion):
            # Construct a PerplexityContent object from the response
            citations = (
                result.search_results if hasattr(result, "search_results") else None
            )
            # Handle potential None or empty citations
            if not citations:
                citations = []

            citations_objs = [PerplexityCitation(**citation) for citation in citations]

            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations_objs
            )
            return content, usage

        return result, usage


class PerplexityClientAsync(PerplexityClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface
        for doing function calling and working with pydantic objects.
        """
        # Keep both raw and instructor clients
        self._raw_client = AsyncOpenAI(
            api_key=self._get_api_key(), base_url="https://api.perplexity.ai"
        )
        return instructor.from_perplexity(self._raw_client)

    @override
    async def query(self, request: Request) -> ConduitResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Now, make the call
        structured_response = None
        if request.response_model is not None:
            # Use instructor for structured responses
            (
                structured_response,
                result,
            ) = await self._client.chat.completions.create_with_completion(
                response_model=request.response_model, **payload_dict
            )
        else:
            # Use raw client for unstructured responses to access citations
            result = await self._raw_client.chat.completions.create(**payload_dict)

        # Capture usage
        if isinstance(result, OpenAIAsyncStream):
            # Handle streaming response if needed; usage is handled by StreamParser
            usage = Usage(input_tokens=0, output_tokens=0)
            return result, usage

        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )

        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage

        if isinstance(result, ChatCompletion):
            # Construct a PerplexityContent object from the response
            citations = (
                result.search_results if hasattr(result, "search_results") else None
            )
            if not citations:
                citations = []

            citations_objs = [PerplexityCitation(**citation) for citation in citations]
            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations_objs
            )
            return content, usage

        return result, usage
