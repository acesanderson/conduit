"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
You want both the 'content' and 'citations' fields from the response object.
Perplexity inputs the footnotes within the content.
For this reason, we define a Pydantic class as our framework is fine with BaseModels as a response. You can still access it as a string and get the content if needed. Citations can be access by choice.
"""

from conduit.model.clients.client import Client, Usage
from conduit.model.clients.load_env import load_env
from conduit.model.clients.perplexity_content import (
    PerplexityContent,
    PerplexityCitation,
)
from conduit.request.request import Request
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
import instructor, tiktoken


class PerplexityClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    def _get_api_key(self):
        api_key = load_env("PERPLEXITY_API_KEY")
        return api_key

    def tokenize(self, model: str, text: str) -> int:
        """
        Return the token count for a string, per model's tokenization function.
        cl100k_base is good enough for Perplexity, per Perplexity documentation.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
        return token_count


class PerplexityClientSync(PerplexityClient):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        # Keep both raw and instructor clients
        self._raw_client = OpenAI(
            api_key=self._get_api_key(), base_url="https://api.perplexity.ai"
        )
        return instructor.from_perplexity(self._raw_client)

    def query(self, request: Request) -> tuple:
        structured_response = None
        if request.response_model is not None:
            # Use instructor for structured responses
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_perplexity()
                )
            )
            # Handle structured response...
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
            assert isinstance(citations, list) and all(
                [isinstance(citation, dict) for citation in citations]
            ), "Citations should be a list of dicts"
            citations = [PerplexityCitation(**citation) for citation in citations]
            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations
            )
            return content, usage
        if isinstance(result, BaseModel):
            return result, usage
        else:
            raise ValueError("Unexpected result type: {}".format(type(result)))


class PerplexityClientAsync(PerplexityClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface
        for doing function calling and working with pydantic objects.
        """
        # Keep both raw and instructor clients
        self._raw_client = AsyncOpenAI(
            api_key=self._get_api_key(), base_url="https://api.perplexity.ai"
        )
        return instructor.from_perplexity(self._raw_client)

    async def query(self, request: Request) -> tuple:
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
            assert isinstance(citations, list) and all(
                [isinstance(citation, dict) for citation in citations]
            ), "Citations should be a list of dicts"
            citations = [PerplexityCitation(**citation) for citation in citations]
            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations
            )
            return content, usage
        if isinstance(result, BaseModel):
            return result, usage
        else:
            raise ValueError("Unexpected result type: {}".format(type(result)))
