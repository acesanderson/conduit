from Chain.message.message import Message
from Chain.progress.wrappers import progress_display
from Chain.progress.verbosity import Verbosity
from Chain.request.request import Request
from Chain.result.result import ChainResult
from Chain.result.error import ChainError
from Chain.logs.logging_config import get_logger
from Chain.request.outputtype import OutputType
from Chain.odometer.OdometerRegistry import OdometerRegistry
from pydantic import ValidationError, BaseModel
from typing import Optional, TYPE_CHECKING
from pathlib import Path
from time import time
import importlib

# Load only if type checking
if TYPE_CHECKING:
    from openai import Stream
    from anthropic import Stream as AnthropicStream
    from rich.console import Console
    from pydantic import BaseModel
    from Chain.cache.cache import ChainCache
    from Chain.model.model_client import ModelClient


# Constants
dir_path = Path(__file__).resolve().parent
logger = get_logger(__name__)


class Model:
    # Class singletons
    _clients = {}  # Store lazy-loaded client instances at the class level
    _chain_cache: Optional["ChainCache"] = (
        None  # If you want to add a cache, add it at class level as a singleton.
    )
    _console: Optional["Console"] = (
        None  # For rich console output, if needed. This is overridden in the Chain class.
    )
    _odometer_registry = OdometerRegistry()

    @classmethod
    def from_server(cls, model: str) -> "ModelClient":
        """
        Factory method to create a ClientModel instance for interacting with a remote server.

        Usage:
        model = Model.from_server("gpt-oss:latest")
        """
        from Chain.model.model_client import ModelClient

        return ModelClient(model)

    @classmethod
    def models(cls) -> dict:
        """
        Returns a dictionary of available models.
        This is useful for introspection and debugging.
        """
        from Chain.model.models.modelstore import ModelStore

        return ModelStore.models()

    @classmethod
    def validate_model(cls, model: str) -> bool:
        """
        Validates the model name against the available models.
        Raises ValueError if the model is not found.
        """
        from Chain.model.models.modelstore import ModelStore

        model_name = ModelStore._validate_model(model)
        if model_name:
            return True
        else:
            return False

    # Object methods
    def __init__(self, model: str = "gpt-4o", console: Optional["Console"] = None):
        from Chain.model.models.modelstore import ModelStore

        self.model = ModelStore._validate_model(model)
        self._client_type = self._get_client_type(self.model)
        self._client = self.__class__._get_client(self._client_type)
        self._console = console

    @classmethod
    def stats(cls):
        """
        Pretty prints session statistics (from OdometerRegistry.session_odometer).
        """
        cls._odometer_registry.session_odometer.stats()

    @property
    def console(self):
        """
        Returns the effective console (hierarchy: instance -> Model class -> Chain/AsyncChain class -> None)
        """
        if self._console:
            return self._console

        import sys

        # Check for Model._console
        if "Chain.model.model" in sys.modules:
            Model = sys.modules["Chain.model.model"].Model
            model_console = getattr(Model, "_console", None)
            if model_console:
                return model_console

        # Check for Chain._console
        if "Chain.chain.chain" in sys.modules:
            Chain = sys.modules["Chain.chain.chain"].Chain
            chain_console = getattr(Chain, "_console", None)
            if chain_console:
                return chain_console

        # Check for AsyncChain._console
        if "Chain.chain.asyncchain" in sys.modules:
            AsyncChain = sys.modules["Chain.chain.asyncchain"].AsyncChain
            async_console = getattr(AsyncChain, "_console", None)
            if async_console:
                return async_console

        return None

    @console.setter
    def console(self, console: "Console"):
        """
        Sets the console object for rich output.
        This is useful if you want to override the default console for a specific instance.
        """
        self._console = console

    def _get_client_type(self, model: str) -> tuple:
        """
        Setting client_type for Model object is necessary for loading the correct client in the query functions.
        Returns a tuple with client type (which informs the module title) and the client class name (which is used to instantiate the client).
        """
        from Chain.model.models.modelstore import ModelStore

        model_list = ModelStore.models()
        if model in model_list["openai"]:
            return "openai", "OpenAIClientSync"
        elif model in model_list["anthropic"]:
            return "anthropic", "AnthropicClientSync"
        elif model in model_list["google"]:
            return "google", "GoogleClientSync"
        elif model in model_list["ollama"]:
            return "ollama", "OllamaClientSync"
        elif model in model_list["groq"]:
            return "groq", "GroqClientSync"
        elif model in model_list["perplexity"]:
            return "perplexity", "PerplexityClientSync"
        elif model in model_list["huggingface"]:
            return "huggingface", "HuggingFaceClientSync"
        else:
            raise ValueError(f"Model {model} not found in models")

    @classmethod
    def _get_client(cls, client_type: tuple):
        # print(f"client type: {client_type}")
        if client_type[0] not in cls._clients:
            try:
                module = importlib.import_module(
                    f"Chain.model.clients.{client_type[0].lower()}_client"
                )
                client_class = getattr(module, f"{client_type[1]}")
                cls._clients[client_type[0]] = client_class()
            except ImportError as e:
                raise ImportError(f"Failed to import {client_type} client: {str(e)}")
        client_object = cls._clients[client_type[0]]
        if not client_object:
            raise ValueError(f"Client {client_type} not found in clients")
        return client_object

    @progress_display
    def query(
        self,
        # Standard parameters
        query_input: str | list | Message | None = None,
        response_model: type["BaseModel"] | None = None,
        cache=True,
        temperature: Optional[float] = None,
        stream: bool = False,
        output_type: OutputType = "text",
        max_tokens: Optional[int] = None,
        # For progress reporting decorator
        verbose: Verbosity = Verbosity.PROGRESS,
        index: int = 0,
        total: int = 0,
        # If we're hand-constructing Request params, we can pass them in directly
        request: Optional[Request] = None,
        # Options for debugging
        return_request: bool = False,
        return_error: bool = False,
    ) -> "ChainResult | Request | Stream | AnthropicStream":
        try:
            # Construct Request object if not provided (majority of cases)
            if not request:
                logger.info(
                    "Constructing Request object from query_input and other parameters."
                )
                import inspect

                frame = inspect.currentframe()
                args, _, _, values = inspect.getargvalues(frame)

                query_args = {k: values[k] for k in args if k != "self"}
                query_args["model"] = self.model
                cache = query_args.pop("cache", False)
                if query_input:
                    query_args.pop("query_input", None)
                    request = Request.from_query_input(
                        query_input=query_input, **query_args
                    )
                else:
                    request = Request(**query_args)

            assert isinstance(request, Request), (
                f"Request must be an instance of Request or None, got {type(request)}"
            )

            # For debug, return Request if requested
            if return_request:
                return request
            # For debug, return error if requested
            if return_error:
                from Chain.tests.fixtures import sample_error

                return sample_error

            # Check cache first
            logger.info("Checking cache for existing results.")
            if cache and self._chain_cache:
                cached_result = self._chain_cache.check_for_model(request)
                if isinstance(cached_result, ChainResult):
                    return (
                        cached_result  # This should be a Response (part of ChainResult)
                    )
                elif cached_result == None:
                    logger.info("No cached result found, proceeding with query.")
                    pass
                elif cached_result and not isinstance(cached_result, ChainResult):
                    logger.error(
                        f"Cache returned a non-ChainResult type: {type(cached_result)}. Ensure the cache is properly configured."
                    )
                    raise ValueError(
                        f"Cache returned a non-ChainResult type: {type(cached_result)}. Ensure the cache is properly configured."
                    )
            # Execute the query
            logger.info("Executing query with client.")
            start_time = time()
            result, usage = self._client.query(request)
            stop_time = time()
            logger.info(f"Query executed in {stop_time - start_time:.2f} seconds.")

            # Handle streaming responses
            from Chain.model.clients.openai_client import Stream
            from Chain.model.clients.anthropic_client import Stream as AnthropicStream

            if isinstance(result, Stream) or isinstance(result, AnthropicStream):
                if stream:
                    logger.info("Returning streaming response.")
                    return result  # Return stream directly
                else:
                    logger.error(
                        "Streaming responses are not supported in this method. "
                        "Set stream=True to receive streamed responses."
                    )
                    raise ValueError(
                        "Streaming responses are not supported in this method. "
                        "Set stream=True to receive streamed responses."
                    )

            # Construct Response object
            from Chain.result.response import Response
            from pydantic import BaseModel

            if isinstance(result, Response):
                logger.info("Returning existing Response object.")
                response = result
            elif isinstance(result, str) or isinstance(result, BaseModel):
                logger.info(
                    "Constructing Response object from result string or BaseModel."
                )
                # Construct relevant Message type per requested output_type
                match output_type:
                    case "text":  # result is a string
                        from Chain.message.textmessage import TextMessage

                        assistant_message = TextMessage(
                            role="assistant", content=result
                        )
                    case "image":  # result is a base64 string of an image
                        assert isinstance(result, str), (
                            "Image generation request should not return a BaseModel; we need base64 string."
                        )
                        from Chain.message.imagemessage import ImageMessage

                        assistant_message = ImageMessage.from_base64(
                            role="assistant", text_content="", image_content=result
                        )
                    case "audio":  # result is a base64 string of an audio
                        assert isinstance(result, str), (
                            "Audio generation (TTS) request should not return a BaseModel; we need base64 string."
                        )
                        from Chain.message.audiomessage import AudioMessage

                        assistant_message = AudioMessage.from_base64(
                            role="assistant",
                            audio_content=result,
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
            else:
                logger.error(
                    f"Unexpected result type: {type(result)}. Expected Response or str."
                )
                raise TypeError(
                    f"Unexpected result type: {type(result)}. Expected Response or str."
                )

            # Update cache after successful query
            logger.info("Updating cache with the new response.")
            if cache and self._chain_cache:
                self._chain_cache.store_for_model(request, response)

            return response  # Return Response (part of ChainResult)

        except ValidationError as e:
            chainerror = ChainError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Validation error: {chainerror}")
            return chainerror
        except Exception as e:
            chainerror = ChainError.from_exception(
                e,
                code="query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Error during query: {chainerror}")
            return chainerror

    def tokenize(self, text: str) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return self._client.tokenize(model=self.model, text=text)

    def pretty(self, user_input):
        pretty = user_input.replace("\n", " ").replace("\t", " ").strip()
        return pretty[:60] + "..." if len(pretty) > 60 else pretty

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
