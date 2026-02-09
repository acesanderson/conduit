"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
We treat citations as metadata attached to the message, ensuring the 'content' remains standard text.
"""

from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.perplexity.payload import PerplexityPayload
from conduit.core.clients.perplexity.message_adapter import (
    convert_message_to_perplexity,
)
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage
from openai import AsyncOpenAI, AsyncStream
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
import os
import time
import json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message


class PerplexityClient(Client):
    """
    Client implementation for Perplexity API using OpenAI-compatible endpoint.
    Async only.
    """

    def __init__(self):
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client

    def _initialize_client(self) -> tuple[Instructor, AsyncOpenAI]:
        """
        Creates both raw and instructor-wrapped clients.
        Raw client for standard completions, Instructor for structured responses.
        Uses instructor.from_perplexity() for Perplexity-specific handling.
        """
        raw_client = AsyncOpenAI(
            api_key=self._get_api_key(),
            base_url="https://api.perplexity.ai",
        )
        instructor_client = instructor.from_perplexity(raw_client)
        return instructor_client, raw_client

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
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Perplexity's SDK (OpenAI-compatible).

        Perplexity-specific parameters:
        - return_citations: Enable citation return
        - return_images: Enable image return
        - search_recency_filter: Time-based filtering
        """
        # load client_params
        client_params = request.params.client_params or {}
        allowed_params = {
            "return_citations",
            "search_recency_filter",
            "search_domain_filter",
        }
        for param in client_params.keys():
            if param not in allowed_params:
                raise ValueError(
                    f"Invalid client_param '{param}' for PerplexityClient. Allowed params: {allowed_params}"
                )
        # convert messages
        converted_messages = self._convert_messages(request.messages)
        # build payload
        perplexity_payload = PerplexityPayload(
            model=request.params.model,
            messages=converted_messages,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            # Perplexity-specific params
            **client_params,
        )
        return perplexity_payload

    @override
    async def tokenize(self, model: str, payload: str | Sequence[Message]) -> int:
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
            # Standard OpenAI-compatible overhead approximation
            tokens_per_message = 3
            num_tokens = 0

            for message in payload:
                num_tokens += tokens_per_message

                # Role
                num_tokens += len(encoding.encode(message.role.value))

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
                        # Extract text parts only to be safe
                        texts = []
                        for block in message.content:
                            if isinstance(block, str):
                                texts.append(block)
                            elif hasattr(block, "text"):
                                texts.append(block.text)
                        content_str = " ".join(texts)
                    except Exception:
                        content_str = str(message.content)
                else:
                    content_str = str(message.content)

                num_tokens += len(encoding.encode(content_str))

            num_tokens += 3  # reply primer
            return num_tokens
        raise ValueError("Payload must be string or Sequence[Message]")

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
        Generate text using Perplexity and return a GenerationResponse.
        Extracts citations from search_results and stores them in MESSAGE METADATA.

        Returns:
            - GenerationResponse object for successful non-streaming requests
            - AsyncStream object for streaming requests
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # The OpenAI SDK rejects these if passed as top-level kwargs.
        pplx_params = {
            "return_citations",
            "return_images",
            "return_related_questions",
            "search_recency_filter",
            "search_domain_filter",
        }

        extra_body = {}
        for param in list(payload_dict.keys()):
            if param in pplx_params:
                extra_body[param] = payload_dict.pop(param)

        if extra_body:
            payload_dict["extra_body"] = extra_body

        # Track timing
        start_time = time.time()

        # Use the raw client for standard completions (to access citations)
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

        # Extract citations from Perplexity response
        citations_raw = (
            result.search_results if hasattr(result, "search_results") else []
        )

        # Normalize citations for metadata
        citations = [
            {
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "source": c.get("source", ""),
                "date": c.get("date", ""),
            }
            for c in citations_raw
        ]

        # Create AssistantMessage
        # CRITICAL FIX: content is just the string. Citations go to metadata.
        assistant_message = AssistantMessage(
            content=result.choices[0].message.content,
            metadata={"citations": citations, "provider": "perplexity"},
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

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate a structured response using Perplexity's function calling.
        Also extracts citations from search_results into metadata.

        Returns:
            - GenerationResponse object with parsed structured data in AssistantMessage.parsed
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # The OpenAI SDK rejects these if passed as top-level kwargs.
        pplx_params = {
            "return_citations",
            "return_images",
            "return_related_questions",
            "search_recency_filter",
            "search_domain_filter",
        }

        extra_body = {}
        for param in list(payload_dict.keys()):
            if param in pplx_params:
                extra_body[param] = payload_dict.pop(param)

        if extra_body:
            payload_dict["extra_body"] = extra_body

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

        # Extract citations from Perplexity response
        citations_raw = (
            completion.search_results if hasattr(completion, "search_results") else []
        )

        citations = [
            {
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "source": c.get("source", ""),
                "date": c.get("date", ""),
            }
            for c in citations_raw
        ]

        # Create AssistantMessage with parsed structured data
        # Citations stored in metadata to avoid polluting the content or parsed object
        assistant_message = AssistantMessage(
            content=completion.choices[0].message.content,
            parsed=user_obj,
            metadata={"citations": citations, "provider": "perplexity"},
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
