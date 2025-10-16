"""
Client subclass for Ollama models.
This doesn't require an API key since these are locally hosted models.
We can use openai api calls to the ollama server, but we use the instructor library to handle the API calls.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
We define preferred defaults for context sizes in a separate json file.
"""

from conduit.model.clients.client import Client, Usage
from conduit.request.request import Request
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI, Stream
from xdg_base_dirs import xdg_state_home, xdg_config_home
from pathlib import Path
from collections import defaultdict
import instructor, ollama, json


DIR_PATH = Path(__file__).resolve().parent
OLLAMA_MODELS_PATH = xdg_state_home() / "conduit" / "ollama_models.json"
OLLAMA_CONTEXT_SIZES_PATH = xdg_config_home() / "conduit" / "ollama_context_sizes.json"


class OllamaClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    # Load Ollama context sizes from the JSON file
    with open(OLLAMA_CONTEXT_SIZES_PATH) as f:
        _ollama_context_data = json.load(f)

    # Use defaultdict to set default context size to 4096 if not specified
    _ollama_context_sizes = defaultdict(lambda: 32768)
    _ollama_context_sizes.update(_ollama_context_data)

    def __init__(self):
        self._client = self._initialize_client()
        self.update_ollama_models()  # This allows us to keep the model file up to date.

    def _initialize_client(self):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    def _get_api_key(self):
        """
        Best thing about Ollama; no API key needed.
        """
        return ""

    def update_ollama_models(self):
        """
        Updates the list of Ollama models.
        We run is every time ollama is initialized.
        """
        # Lazy load ollama module
        ollama_models = [m["model"] for m in ollama.list()["models"]]
        ollama_model_dict = {"ollama": ollama_models}
        with open(OLLAMA_MODELS_PATH, "w") as f:
            json.dump(ollama_model_dict, f)

    def tokenize(self, model: str, text: str) -> int:
        """
        Count tokens using Ollama's generate API via the official library.
        This actually runs a text generation, but only only for one token to minimize compute, since we only want the count of input tokens.
        """
        response = ollama.generate(
            model=model,
            prompt=text,
            options={"num_predict": 1},  # Set to minimal generation
        )
        return int(response.get("prompt_eval_count", 0))


class OllamaClientSync(OllamaClient):
    def _initialize_client(self):
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    def query(
        self,
        request: Request,
    ) -> tuple:
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
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        # Try to retrieve the text first
        try:
            return result.choices[0].message.content, usage
        except AttributeError:
            # If the result is not in the expected format, return the raw result
            pass
        if isinstance(result, BaseModel):
            return result, usage
        if isinstance(result, Stream):
            # Handle streaming response if needed
            return result, usage


class OllamaClientAsync(OllamaClient):
    def _initialize_client(self):
        """
        This is just ollama's async client.
        """
        ollama_async_client = instructor.from_openai(
            AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
        return ollama_async_client

    async def query(
        self,
        request: Request,
    ) -> tuple:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Ollama, so we use `create_with_completion`
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
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        # Try to retrieve the text first
        try:
            return result.choices[0].message.content, usage
        except AttributeError:
            # If the result is not in the expected format, return the raw result
            pass
        if isinstance(result, BaseModel):
            return result, usage
