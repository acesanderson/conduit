"""
Client subclass for Anthropic models.
TBD: implement streaming support.
"""

from conduit.model.clients.client import Client, Usage
from conduit.request.request import Request
from conduit.model.clients.load_env import load_env
from anthropic import Anthropic, AsyncAnthropic, Stream
from pydantic import BaseModel
import instructor
import os


class AnthropicClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

    def _get_api_key(self):
        load_env("ANTHROPIC_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY") is None:
            raise ValueError("No ANTHROPIC_API_KEY found in environment variables")
        else:
            return os.getenv("ANTHROPIC_API_KEY")

    def tokenize(self, model: str, text: str) -> int:
        """
        Get token count per official Anthropic api endpoint.
        """
        # Convert text to message format
        anthropic_client = Anthropic(api_key=self._get_api_key())
        messages = [{"role": "user", "content": text}]
        token_count = anthropic_client.messages.count_tokens(
            model=model,
            messages=messages,
        )
        return token_count.input_tokens


class AnthropicClientSync(AnthropicClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

    def query(
        self,
        request: Request,
    ) -> tuple:
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
        elif isinstance(result, Stream):
            # Handle streaming response if needed
            return result, usage
        else:
            raise ValueError(
                f"Unexpected result type from Anthropic API: {type(result)}"
            )


class AnthropicClientAsync(AnthropicClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_async_client = AsyncAnthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_async_client)

    async def query(
        self,
        request: Request,
    ) -> tuple:
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
