"""
For Google Gemini models.
"""

from conduit.model.clients.client import Client, Usage
from conduit.model.clients.load_env import load_env
from conduit.request.request import Request
from openai import OpenAI, AsyncOpenAI, Stream
import instructor
from pydantic import BaseModel


class GoogleClient(Client):
    """
    This is a base class; we have two subclasses: GoogleClientSync and GoogleClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _get_api_key(self):
        api_key = load_env("GOOGLE_API_KEY")
        return api_key

    def tokenize(self, model: str, text: str) -> int:
        """
        Get the token count per official tokenizer (through API).
        """
        # Example using google-generativeai SDK for estimation
        import google.generativeai as genai

        client = genai.GenerativeModel(model_name=model)
        response = client.count_tokens(text)
        token_count = response.total_tokens
        return token_count


class GoogleClientSync(GoogleClient):
    def _initialize_client(self):
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

    def query(
        self,
        request: Request,
    ) -> tuple:
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
    def _initialize_client(self):
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

    async def query(
        self,
        request: Request,
    ) -> tuple:
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
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            pass
        if isinstance(result, BaseModel):
            return result, usage
