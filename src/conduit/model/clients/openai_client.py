from __future__ import annotations
from conduit.model.clients.client import Client, Usage
from conduit.model.clients.load_env import load_env
from openai import OpenAI
from openai import AsyncOpenAI
from openai import Stream
from abc import ABC
import instructor
from instructor import Instructor
import logging
import json
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from conduit.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.request.request import Request
    from conduit.message.message import Message

logger = logging.getLogger(__name__)


class OpenAIClient(Client, ABC):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    """

    def __init__(self):
        self._client: Instructor = self._initialize_client()

    @override
    def _get_api_key(self) -> str:
        api_key = load_env("OPENAI_API_KEY")
        return api_key

    @override
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Return the token count for a string or a message list.
        For list[Message], it calculates the overhead per OpenAI ChatML format.
        """
        encoding = self._get_encoding(model)

        if isinstance(payload, str):
            return len(encoding.encode(payload))

        if isinstance(payload, list):
            # Lazy import to avoid circular dependency
            from conduit.message.textmessage import TextMessage
            from conduit.message.imagemessage import ImageMessage
            from conduit.message.audiomessage import AudioMessage

            tokens_per_message = 3
            num_tokens = 0

            for message in payload:
                num_tokens += tokens_per_message

                # Role
                num_tokens += len(encoding.encode(message.role))

                # Content
                if isinstance(message, TextMessage):
                    content_str = str(message.content)
                    if not isinstance(message.content, str):
                        try:
                            if isinstance(message.content, list):
                                content_str = json.dumps(
                                    [m.model_dump() for m in message.content]
                                )
                            else:
                                content_str = message.content.model_dump_json()
                        except AttributeError:
                            pass
                    num_tokens += len(encoding.encode(content_str))

                elif isinstance(message, ImageMessage):
                    # 1. Text prompt
                    num_tokens += len(encoding.encode(message.text_content))
                    # 2. Vision buffer (Safety heuristic since tiktoken can't see images)
                    num_tokens += 1000

                elif isinstance(message, AudioMessage):
                    # 1. Text prompt
                    num_tokens += len(encoding.encode(message.text_content))
                    # Audio data is excluded from text context window in standard accounting

            num_tokens += 3  # reply primer
            return num_tokens

        raise ValueError("Payload must be string or list[Message]")

    def _get_encoding(self, model: str):
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def discover_models(self) -> list[str]:
        import openai

        API_KEY = self._get_api_key()
        openai.api_key = API_KEY
        models = openai.models.list()

        return [model.id for model in models.data]


class OpenAIClientSync(OpenAIClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_client = OpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_client)

    @override
    def query(
        self,
        request: Request,
    ) -> tuple[str | object | SyncStream, Usage]:
        match request.output_type:
            case "text":
                return self._generate_text(request)
            case "image":
                return self._generate_image(request)
            case "audio":
                return self._generate_audio(request)
            case _:
                raise ValueError(f"Unsupported output type: {request.output_type}")

    def _generate_text(
        self, request: Request
    ) -> tuple[str | object | SyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_openai()
                )
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(**request.to_openai())
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
            # TBD: whether we handle STREAMING structured responses
            return structured_response, usage
        # First try to get text content from the result
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            # If the result is a BaseModel or Stream, handle accordingly
            pass
        return result, usage

    def _generate_image(self, request: Request) -> tuple[str, Usage]:
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

    def _generate_audio(self, request: Request) -> tuple[str, Usage]:
        raise NotImplementedError


class OpenAIClientAsync(OpenAIClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_async_client = AsyncOpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_async_client)

    @override
    async def query(
        self,
        request: Request,
    ) -> tuple[str | object | AsyncStream, Usage]:
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
        result = result.choices[0].message.content
        return result, usage
