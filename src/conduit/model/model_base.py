"""
This abstract class handles everything *except* the network call.
* **Responsibilities:**
    * Input preparation (Request construction).
    * Input validation (Token counting, parameter checking).
    * Cache Logic (Check cache key).
    * Response Processing (Standardizing raw API output).
    * Post-processing (Saving to cache, emitting Odometer events).
* **State:** Holds the `conduit_cache` and `console` defaults (allowing instance overrides).
"""

from __future__ import annotations
from conduit.config import settings
from conduit.request.request import Request
from conduit.request.query import QueryInput
from conduit.model.clients.client import Usage, Client
from conduit.parser.stream.protocol import SyncStream, AsyncStream
from conduit.result.response import Response
from conduit.result.result import ConduitResult
from conduit.progress.verbosity import Verbosity
from conduit.odometer.OdometerRegistry import OdometerRegistry
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod
import logging

# Load only if type checking
if TYPE_CHECKING:
    from rich.console import Console
    from conduit.cache.cache import ConduitCache

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    Stem class for Model implementations; not to be used directly.
    """

    # Class singletons
    conduit_cache: ConduitCache | None = (
        None  # If you want to add a cache, add it at class level as a singleton.
    )
    _console: Console | None = (
        None  # For rich console output, if needed. This is overridden in the Conduit class.
    )
    _odometer_registry = OdometerRegistry()

    def __init__(
        self,
        model: str = settings.preferred_model,
        console: Console | None = None,
        verbosity: Verbosity = settings.default_verbosity,
    ):
        from conduit.model.models.modelstore import ModelStore

        self.name: str = ModelStore.validate_model(model)
        self.verbosity: Verbosity = verbosity
        self.client: Client = self.get_client(self.name)
        self._console = console

    @classmethod
    def models(cls) -> dict[str, list[str]]:
        """
        Returns a dictionary of available models.
        This is useful for introspection and debugging.
        """
        from conduit.model.models.modelstore import ModelStore

        return ModelStore.models()

    @classmethod
    def stats(cls):
        """
        Pretty prints session statistics (from OdometerRegistry.session_odometer).
        """
        cls._odometer_registry.session_odometer.stats()

    @property
    def console(self):
        """
        Returns the effective console (hierarchy: instance -> Model class -> SyncConduit/AsyncConduit class -> None)
        """
        if self._console:
            return self._console

        import sys

        # Check for Model._console
        if "conduit.model.model" in sys.modules:
            Model = sys.modules["conduit.model.model"].Model
            model_console = getattr(Model, "_console", None)
            if model_console:
                return model_console

        # Check for SyncConduit._console
        if "conduit.conduit.sync_conduit" in sys.modules:
            SyncConduit = sys.modules["conduit.conduit.sync_conduit"].SyncConduit
            conduit_console = getattr(SyncConduit, "_console", None)
            if conduit_console:
                return conduit_console

        # Check for AsyncConduit._console
        if "conduit.conduit.async_conduit" in sys.modules:
            AsyncConduit = sys.modules["conduit.conduit.async_conduit"].AsyncConduit
            async_console = getattr(AsyncConduit, "_console", None)
            if async_console:
                return async_console

        return None

    @console.setter
    def console(self, console: Console):
        """
        Sets the console object for rich output.
        This is useful if you want to override the default console for a specific instance.
        """
        self._console = console

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"

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
                    from conduit.message.textmessage import TextMessage

                    assistant_message = TextMessage(
                        role="assistant", content=raw_result
                    )
                case "image":  # result is a base64 string of an image
                    assert isinstance(raw_result, str), (
                        "Image generation request should not return a BaseModel; we need base64 string."
                    )
                    from conduit.message.imagemessage import ImageMessage

                    assistant_message = ImageMessage.from_base64(
                        role="assistant", text_content="", image_content=raw_result
                    )
                case "audio":  # result is a base64 string of an audio
                    assert isinstance(raw_result, str), (
                        "Audio generation (TTS) request should not return a BaseModel; we need base64 string."
                    )
                    from conduit.message.audiomessage import AudioMessage

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
