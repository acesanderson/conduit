"""
For Google Gemini models.
"""

from __future__ import annotations
from conduit.core.model.clients.client import Client, Usage
from openai import OpenAI, AsyncOpenAI, Stream
from typing import TYPE_CHECKING, override
import instructor
from instructor import Instructor
from abc import ABC
from pydantic import BaseModel
import os

if TYPE_CHECKING:
    from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.domain.request.request import Request
    from conduit.domain.message.message import Message


class GoogleClient(Client, ABC):
    """
    This is a base class; we have two subclasses: GoogleClientSync and GoogleClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client: Instructor = self._initialize_client()

    @override
    def _get_api_key(self) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        else:
            return api_key

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


class GoogleClientSync(GoogleClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        client = instructor.from_openai(
            OpenAI(
                api_key=self._get_api_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    @override
    def query(
        self,
        request: Request,
    ) -> tuple[str | SyncStream | BaseModel, Usage]:
        match request.output_type:
            case "text":
                return self._generate_text(request)
            case "image":
                return self._generate_image(request)
            case "audio":
                return self._generate_audio(request)
            case _:
                raise ValueError(f"Unsupported output type: {request.output_type}")

    def _generate_text(self, request: Request) -> tuple:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Google, so we use `create_with_completion`
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_openai()
                )
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(**request.to_openai())
        # Handle streaming response if needed
        if isinstance(result, Stream):
            usage = Usage(input_tokens=0, output_tokens=0)
            return result, usage
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        # First try to get text content from the result
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            pass
        if isinstance(result, BaseModel):
            return result, usage

    def _generate_image(self, request: Request) -> tuple:
        response = self._client.images.generate(
            model=request.model,
            prompt=request.messages[-1].content,
            response_format="b64_json",
            n=1,
        )
        result = response.data[0].b64_json
        assert isinstance(result, str)
        usage = Usage(input_tokens=0, output_tokens=0)
        return result, usage

    def _generate_audio(self, request: Request) -> tuple:
        raise NotImplementedError


class GoogleClientAsync(GoogleClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        client = instructor.from_openai(
            AsyncOpenAI(
                api_key=self._get_api_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    @override
    async def query(
        self,
        request: Request,
    ) -> tuple[str | AsyncStream | BaseModel, Usage]:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            (
                structured_response,
                result,
            ) = await self._client.chat.completions.create_with_completion(
                **request.to_openai()
            )
        else:
            # Use the standard completion method
            result = await self._client.chat.completions.create(**request.to_openai())
        # Handle streaming response if needed
        if isinstance(result, Stream):
            return result, usage
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        # First try to get text content from the result
        result_str = result.choices[0].message.content
        return result_str, usage
