"""
ClientModel - A server-based Model implementation that maintains protocol compatibility with Model class.

Unlike ModelAsync, this does NOT inherit from Model, but implements the same interface.
"""

from Chain.progress.wrappers import progress_display
from Chain.progress.verbosity import Verbosity
from Chain.request.request import Request
from Chain.result.result import ChainResult
from Chain.result.error import ChainError
from Chain.logs.logging_config import get_logger
from Chain.request.outputtype import OutputType
from Chain.message.message import Message
from pydantic import ValidationError, BaseModel
from typing import Optional, TYPE_CHECKING
from time import time

# Load only if type checking
if TYPE_CHECKING:
    from rich.console import Console

logger = get_logger(__name__)


class ModelClient:
    """
    Server-based model client that implements the same protocol as Model class.
    Works seamlessly with Chain and other components.
    """

    def __init__(
        self, model: str = "gpt-oss:latest", console: Optional["Console"] = None
    ):
        """
        Initialize ClientModel with server connection.

        Args:
            model: Model name (must be available on server)
            server_url: Optional server URL (defaults to SiphonClient default)
            console: Optional Rich console for progress display
        """
        self.model = model
        self._console = console
        self._client = self._initialize_client()
        self._validate_server_model()

    def _initialize_client(self):
        """Initialize SiphonClient connection"""
        from SiphonServer.client.siphonclient import SiphonClient

        client = SiphonClient()

        # Test connection
        try:
            status = client.get_status()
            if status.get("status") != "healthy":
                raise ConnectionError("Server is not healthy")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Chain server: {e}")

    def _validate_server_model(self):
        """Validate that the model is available on the server"""
        try:
            status = self._client.get_status()
            available_models = status.get("models_available", [])

            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' not available on server. "
                    f"Available models: {available_models}"
                )
        except Exception as e:
            raise ValueError(f"Failed to validate model on server: {e}")

    @property
    def console(self):
        """
        Returns the effective console (matches Model's console hierarchy)
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

        return None

    @console.setter
    def console(self, console: "Console"):
        """Sets the console object for rich output"""
        self._console = console

    @progress_display
    def query(
        self,
        # Standard parameters (match Model.query signature)
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
        # If we're hand-constructing Request params
        request: Optional[Request] = None,
        # Options for debugging
        return_request: bool = False,
        return_error: bool = False,
    ) -> ChainResult:
        """
        Query the server model - matches Model.query() signature exactly
        """
        try:
            # Construct Request object if not provided
            if not request:
                logger.info("Constructing Request object for server client")
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

            # For debug, return Request if requested
            if return_request:
                return request

            # For debug, return error if requested
            if return_error:
                from Chain.tests.fixtures import sample_error

                return sample_error

            # Server doesn't support streaming
            if stream:
                logger.warning(
                    "Server does not support streaming, ignoring stream=True"
                )

            # Execute the query via server
            logger.info("Sending query to Chain server")
            start_time = time()

            # Use SiphonClient's query_sync method
            response = self._client.query_sync(request)

            stop_time = time()
            logger.info(
                f"Server query completed in {stop_time - start_time:.2f} seconds"
            )

            # The server should return a Response object
            from Chain.result.response import Response

            if isinstance(response, Response):
                return response
            else:
                raise TypeError(f"Expected Response from server, got {type(response)}")

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
                code="server_query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Error during server query: {chainerror}")
            return chainerror

    def tokenize(self, text: str) -> int:
        """
        Get token count - fallback to basic estimation since server may not support this
        """
        try:
            # Try to use server tokenization if available
            # This would need to be implemented on the server side
            return len(text.split()) * 1.3  # Rough estimation
        except Exception:
            # Fallback to word-based estimation
            return int(len(text.split()) * 1.3)

    def __repr__(self):
        """String representation"""
        return f"ClientModel(model='{self.model}', server_url='{self.server_url}')"
