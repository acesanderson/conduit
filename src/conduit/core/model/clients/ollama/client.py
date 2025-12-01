"""
Client subclass for Ollama models.
This doesn't require an API key since these are locally hosted models.
We can use openai api calls to the ollama server, but we use the instructor library to handle the API calls.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
We define preferred defaults for context sizes in a separate json file.
"""

from __future__ import annotations
from conduit.core.model.clients.client import Client, Usage
from conduit.config import settings
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI, Stream
from pathlib import Path
from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, override
import instructor
from instructor import Instructor
import json
import logging

if TYPE_CHECKING:
    from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.domain.request.request import Request
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)

OLLAMA_MODELS_PATH = settings.paths["OLLAMA_MODELS_PATH"]
OLLAMA_CONTEXT_SIZES_PATH = settings.paths["OLLAMA_CONTEXT_SIZES_PATH"]


class OllamaClient(Client, ABC):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    # Load Ollama context sizes from the JSON file
    try:
        Path(OLLAMA_CONTEXT_SIZES_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(OLLAMA_CONTEXT_SIZES_PATH) as f:
            _ollama_context_data = json.load(f)

        # Use defaultdict to set default context size to 4096 if not specified
        _ollama_context_sizes = defaultdict(lambda: 32768)
        _ollama_context_sizes.update(_ollama_context_data)
    except Exception:
        logger.warning(
            f"Could not load Ollama context sizes from {OLLAMA_CONTEXT_SIZES_PATH}. Using default of 32768."
        )
        _ollama_context_sizes = defaultdict(lambda: 32768)

    def __init__(self):
        self._client: Instructor = self._initialize_client()
        self.update_ollama_models()  # This allows us to keep the model file up to date.

    @override
    def _initialize_client(self) -> Instructor:
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    @override
    def _get_api_key(self) -> str:
        """
        Best thing about Ollama; no API key needed.
        """
        return ""

    def update_ollama_models(self):
        """
        Updates the list of Ollama models.
        We run is every time ollama is initialized.
        """
        import ollama

        # Ensure the directory exists
        Path(OLLAMA_MODELS_PATH).parent.mkdir(parents=True, exist_ok=True)
        # Lazy load ollama module
        ollama_models = [m["model"] for m in ollama.list()["models"]]
        ollama_model_dict = {"ollama": ollama_models}
        with open(OLLAMA_MODELS_PATH, "w") as f:
            json.dump(ollama_model_dict, f)

    @override
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Count tokens using Ollama's API via the official library.
        We set "num_predict" to 0 so we only process the prompt/history and get the eval count.
        """
        import ollama

        # CASE 1: Raw String
        if isinstance(payload, str):
            response = ollama.generate(
                model=model,
                prompt=payload,
                options={"num_predict": 0},  # Minimal generation
            )
            return int(response.get("prompt_eval_count", 0))

        # CASE 2: Message History
        if isinstance(payload, list):
            # Convert internal Messages to OpenAI/Ollama compatible dicts
            messages_payload = [m.to_ollama() for m in payload]

            response = ollama.chat(
                model=model,
                messages=messages_payload,
                options={"num_predict": 0},
            )
            return int(response.get("prompt_eval_count", 0))

        raise ValueError("Payload must be string or list[Message]")


class OllamaClientSync(OllamaClient):
    @override
    def _initialize_client(self) -> Instructor:
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    @override
    def query(
        self,
        request: Request,
    ) -> tuple[str | BaseModel | SyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_ollama()
                )
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(**request.to_ollama())
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
        # Try to retrieve the text first
        return result.choices[0].message.content, usage


class OllamaClientAsync(OllamaClient):
    @override
    def _initialize_client(self) -> Instructor:
        """
        This is just ollama's async client.
        """
        ollama_async_client = instructor.from_openai(
            AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
        return ollama_async_client

    @override
    async def query(
        self,
        request: Request,
    ) -> tuple[str | BaseModel | AsyncStream, Usage]:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Ollama, so we use `create_with_completion`
            (
                structured_response,
                result,
            ) = await self._client.chat.completions.create_with_completion(
                **request.to_ollama()
            )
        else:
            # Use the standard completion method
            result = await self._client.chat.completions.create(**request.to_ollama())
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
        # Try to retrieve the text first
        return result.choices[0].message.content, usage
