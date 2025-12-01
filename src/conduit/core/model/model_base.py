"""
### Model vs. Conduit: The Division of Labor

The **Conduit** class is your **Workflow Orchestrator**; it handles the *context* of the application—templating prompts, managing conversation history, and governing the shape of the execution flow. The **Model** class is your **Execution Runtime**; it handles the *mechanics* of intelligence. It is responsible for normalizing disparate provider APIs (OpenAI, Anthropic, Ollama) into a unified standard, managing low-level infrastructure like token accounting and caching, and executing the actual network I/O.

### The Model Family Taxonomy

* **`ModelSync` (Blocking I/O):** The default implementation that executes requests synchronously, blocking the main thread until completion; ideal for linear scripts, CLIs, and simple tools.
* **`ModelAsync` (Non-Blocking I/O):** An asynchronous implementation that returns awaitable coroutines, designed for integration with Python's `asyncio` event loop to support high-concurrency applications.
* **`RemoteModel` (Proxy Execution):** A lightweight client that serializes requests and delegates execution to a centralized `Headwater` server, abstracting away local provider dependencies and API key management.
* **`ModelBase` (The Abstract Stem):** The foundational abstract class that defines the core Request/Response protocol, cache logic, and odometer tracking shared by all execution strategies.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.request.request import Request
from conduit.domain.request.query import QueryInput
from conduit.core.model.clients.client import Usage, Client
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
from conduit.domain.result.response import Response
from conduit.domain.result.result import ConduitResult
from conduit.utils.progress.verbosity import Verbosity
from conduit.storage.odometer.OdometerRegistry import OdometerRegistry
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, override
from abc import ABC, abstractmethod
import logging

# Load only if type checking
if TYPE_CHECKING:
    from rich.console import Console
    from conduit.storage.cache.cache import ConduitCache

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    Stem class for Model implementations; not to be used directly.
    """

    # Class singleton
    _odometer_registry: OdometerRegistry = OdometerRegistry()

    def __init__(
        self,
        model_name: str = settings.preferred_model,
        console: Console | None = settings.default_console,
        verbosity: Verbosity = settings.default_verbosity,
        cache: str | ConduitCache | None = None,
    ):
        from conduit.core.model.models.modelstore import ModelStore

        self.name: str = ModelStore.validate_model(model_name)
        self.verbosity: Verbosity = verbosity
        self.console: Console | None = console
        self.client: Client = self.get_client(self.name)

        if cache is not None:
            from conduit.storage.cache.cache import ConduitCache

            if isinstance(cache, str):
                # Convenience: Create cache from string name

                logger.info(f"Initializing cache with name: '{cache}'")
                self.cache: str | ConduitCache | None = ConduitCache(name=cache)
            else:  # It's already a ConduitCache instance
                self.cache = cache
        else:
            self.cache = None

    # Config methods (post-init)
    def enable_cache(self) -> None:
        if self.cache is None:
            from conduit.storage.cache.cache import ConduitCache

            logger.info("Enabling default cache.")
            self.cache = ConduitCache()

    def enable_console(self) -> None:
        if self.console is None:
            logger.info("Enabling console.")
            self.console = Console()

    def disable_cache(self) -> None:
        if self.cache is not None:
            logger.info("Disabling cache.")
            self.cache = None

    def disable_console(self) -> None:
        if self.console is not None:
            logger.info("Disabling console.")
            self.console = None

    # Class methods for global info
    @classmethod
    def models(cls) -> dict[str, list[str]]:
        """
        Returns a dictionary of available models.
        This is useful for introspection and debugging.
        """
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.models()

    @classmethod
    def stats(cls):
        """
        Pretty prints session statistics (from OdometerRegistry.session_odometer).
        """
        cls._odometer_registry.session_odometer.stats()

    # Query methods: these are orchestrated in subclasses
    def _prepare_request(
        self, query_input: QueryInput | None = None, **kwargs
    ) -> Request:
        """
        PURE CPU: Constructs and validates the Request object.
        """

        request: Request = kwargs.pop("request", None)

        if request is None:
            if query_input is None:
                raise ValueError("query_input is required when no request is provided.")

            # inject defaults
            kwargs.setdefault("model", self.name)

            request: Request = Request.from_query_input(
                query_input=query_input, **kwargs
            )
        else:
            if not isinstance(request, Request):
                raise TypeError(f"request must be a Request, got {type(req)}")

        return request

    def _process_response(
        self,
        raw_result: Any,
        usage: Usage,
        request: Request,
        start_time: float,
        stop_time: float,
    ) -> ConduitResult:
        """
        PURE CPU: Converts raw client output into a standard Response object.
        """
        output_type = request.output_type
        # Streaming responses are returned as-is
        if isinstance(raw_result, (SyncStream, AsyncStream)):
            logger.info("Returning streaming response.")
            return raw_result
        # Non-streaming responses
        if isinstance(raw_result, Response):
            logger.info("Returning existing Response object.")
            response = raw_result
        # Handle string or BaseModel results
        if isinstance(raw_result, (str, BaseModel)):
            logger.info("Constructing Response object from result string or BaseModel.")
            # Construct relevant Message type per requested output_type
            match output_type:
                case "text":  # result is a string
                    from conduit.domain.message.textmessage import TextMessage

                    assistant_message = TextMessage(
                        role="assistant", content=raw_result
                    )
                case "image":  # result is a base64 string of an image
                    assert isinstance(raw_result, str), (
                        "Image generation request should not return a BaseModel; we need base64 string."
                    )
                    from conduit.domain.message.imagemessage import ImageMessage

                    assistant_message = ImageMessage.from_base64(
                        role="assistant", text_content="", image_content=raw_result
                    )
                case "audio":  # result is a base64 string of an audio
                    assert isinstance(raw_result, str), (
                        "Audio generation (TTS) request should not return a BaseModel; we need base64 string."
                    )
                    from conduit.domain.message.audiomessage import AudioMessage

                    assistant_message = AudioMessage.from_base64(
                        role="assistant",
                        audio_content=raw_result,
                        text_content="",
                        format="mp3",
                    )

            response = Response(
                message=assistant_message,
                request=request,
                duration=stop_time - start_time,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            return response
        else:
            raise TypeError(
                f"Unexpected result type: {type(raw_result)}. Expected Response, BaseModel, or str."
            )

    # I/O Helpers (Synchronous by default)
    def _check_cache(self, request: Request) -> Response | None:
        """
        PURE CPU: Checks the cache for existing results. (well, we may wrap in async later)
        """
        if self.conduit_cache:
            cached_result = self.conduit_cache.check_for_model(request)
            if isinstance(cached_result, Response):
                return cached_result
            elif cached_result == None:
                logger.info("No cached result found, proceeding with query.")
                pass
            elif cached_result and not isinstance(cached_result, ConduitResult):
                logger.error(
                    f"Cache returned a non-ConduitResult type: {type(cached_result)}. Ensure the cache is properly configured."
                )
                raise ValueError(
                    f"Cache returned a non-ConduitResult type: {type(cached_result)}. Ensure the cache is properly configured."
                )
        else:
            raise ValueError("No cache configured for this model instance.")

    def _save_cache(self, request: Request, response: Response):
        """
        PURE CPU: Saves the response to the cache. (well, we may wrap in async later)
        """
        if self.conduit_cache:
            self.conduit_cache.store_for_model(request, response)

    # Expected methods in subclasses
    @abstractmethod
    def get_client(self, model_name: str) -> Client: ...

    @abstractmethod
    def query(self, query_input: QueryInput, **kwargs) -> ConduitResult: ...

    @abstractmethod
    def tokenize(self, text: str) -> int: ...

    # Dunders
    @override
    def __repr__(self) -> str:
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
