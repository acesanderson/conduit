"""
For Google Gemini models.
"""

from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.storage.odometer.usage import Usage
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.google.payload import GooglePayload
from conduit.core.clients.google.adapter import convert_message_to_google
from openai import OpenAI, AsyncOpenAI, Stream, AsyncStream
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
from abc import ABC
import os

if TYPE_CHECKING:
    from conduit.domain.result.result import ConduitResult
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
        return api_key

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into Google's specific dictionary format.
        Since Google uses OpenAI spec, we delegate to the OpenAI adapter.
        """
        return convert_message_to_google(message)

    @override
    def _convert_request(self, request: Request) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Google's SDK (via OpenAI spec).
        """
        # Load client params
        client_params = request.client_params or {}
        allowed_params = {"frequency_penalty", "presence_penalty"}
        for param in client_params.keys():
            if param not in allowed_params:
                raise ValueError(f"Unsupported Google client parameter: {param}")
        # Convert messages
        converted_messages = self._convert_messages(request.messages)
        # Build payload
        google_payload = GooglePayload(
            model=request.model,
            messages=converted_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream,
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


class GoogleClientSync(GoogleClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        client = instructor.from_openai(
            OpenAI(
                api_key=self._get_api_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    @override
    def query(
        self,
        request: Request,
    ) -> ConduitResult:
        match request.output_type:
            case "text":
                return self._generate_text(request)
            case "image":
                return self._generate_image(request)
            case "audio":
                return self._generate_audio(request)
            case _:
                raise ValueError(f"Unsupported output type: {request.output_type}")

    def _generate_text(self, request: Request) -> ConduitResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)
        # Now, make the call
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Google, so we use `create_with_completion`
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    response_model=request.response_model, **payload_dict
                )
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(
                response_model=request.response_model, **payload_dict
            )
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
        # First try to get text content from the result
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            # If the result is a BaseModel or Stream, handle accordingly
            pass
        return result, usage

    def _generate_image(self, request: Request) -> ConduitResult:
        response = self._client.images.generate(
            model=request.model,
            prompt=request.messages[-1].content,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        result = response.data[0].b64_json
        assert isinstance(result, str)
        usage = Usage(input_tokens=0, output_tokens=0)
        return result, usage

    def _generate_audio(self, request: Request) -> ConduitResult:
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
                base_url="https://generativelanguage.googleapis.com/v1beta/",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

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
            # We want the raw response from Google, so we use `create_with_completion`
            (
                structured_response,
                result,
            ) = await self._client.chat.completions.create_with_completion(
                response_model=request.response_model, **payload_dict
            )
        else:
            # Use the standard completion method
            result = await self._client.chat.completions.create(
                response_model=request.response_model, **payload_dict
            )
        # Capture usage
        if isinstance(result, AsyncStream):
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
        result = result.choices[0].message.content
        return result, usage
