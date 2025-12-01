"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
You want both the 'content' and 'citations' fields from the response object.
"""

from __future__ import annotations
from conduit.core.model.clients.client import Client, Usage
from conduit.core.model.clients.perplexity_content import (
    PerplexityContent,
    PerplexityCitation,
)
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from abc import ABC
from typing import TYPE_CHECKING, override
import instructor
from instructor import Instructor
import json
import os

if TYPE_CHECKING:
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
    def query(
        self, request: Request
    ) -> tuple[str | BaseModel | PerplexityContent | SyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            # Use instructor for structured responses
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_perplexity()
                )
            )
        else:
            # Use raw client for unstructured responses
            perplexity_params = request.to_perplexity()
            perplexity_params.pop("response_model", None)  # Remove None response_model
            result = self._raw_client.chat.completions.create(**perplexity_params)

        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )

        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage

        if isinstance(result, ChatCompletion):
            # Construct a PerplexityContent object from the response
            citations = result.search_results
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
    async def query(
        self, request: Request
    ) -> tuple[str | BaseModel | PerplexityContent | AsyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            # Use instructor for structured responses
            (
                structured_response,
                result,
            ) = await self._client.chat.completions.create_with_completion(
                **request.to_perplexity()
            )
        else:
            # Use raw client for unstructured responses
            perplexity_params = request.to_perplexity()
            perplexity_params.pop("response_model", None)  # Remove None response_model
            result = await self._raw_client.chat.completions.create(**perplexity_params)

        # Capture usage
        if hasattr(result, "usage"):
            # Raw OpenAI/Perplexity response
            usage = Usage(
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
            )
        else:
            # Instructor wrapped response - need to get usage from _raw_response
            raw_response = result._raw_response  # instructor stores original here
            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
            )

        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage

        if isinstance(result, ChatCompletion):
            # Construct a PerplexityContent object from the response
            citations = result.search_results
            if not citations:
                citations = []

            citations_objs = [PerplexityCitation(**citation) for citation in citations]
            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations_objs
            )
            return content, usage

        return result, usage
