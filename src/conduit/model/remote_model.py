"""
RemoteModel - A server-based Model implementation that maintains protocol compatibility with Model class.

Unlike ModelAsync, this does NOT inherit from Model, but implements the same interface.
"""

from headwater_client.client.headwater_client import HeadwaterClient
from headwater_api.classes import StatusResponse
from conduit.progress.wrappers import progress_display
from conduit.progress.verbosity import Verbosity
from conduit.request.request import Request
from conduit.result.result import ConduitResult
from conduit.result.error import ConduitError
from conduit.request.outputtype import OutputType
from conduit.message.message import Message
from pydantic import ValidationError, BaseModel
from typing import TYPE_CHECKING
from time import time
import logging

# Load only if type checking
if TYPE_CHECKING:
    from rich.console import Console


logger = logging.getLogger(__name__)


class RemoteModel:
    """
    Server-based model client that implements the same protocol as Model class.
    Works seamlessly with Conduit and other components.
    """

    def __init__(self, model: str = "gpt-oss:latest", console: "Console | None" = None):
        """
        Initialize RemoteModel with server connection.

        Args:
            model: Model name (must be available on server)
            console: Optional Rich console for progress display
        """
        self.model: str = model
        self._console: Console = console
        self._client: HeadwaterClient = self._initialize_client()
        self._validate_server_model()

    def _initialize_client(self) -> HeadwaterClient:
        """Initialize SiphonClient connection"""
        client = HeadwaterClient()

        # Test connection
        try:
            status = client.ping()
            if status == False:
                raise ConnectionError("Server is not healthy")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Headwater server: {e}")

    def _validate_server_model(self):
        """Validate that the model is available on the server"""
        try:
            status: StatusResponse = self._client.get_status()
            available_models = getattr(status, "models_available", [])

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

        # Check for SyncConduit._console
        if "conduit.conduit.sync_conduit" in sys.modules:
            conduit = sys.modules["conduit.conduit.sync_conduit"].SyncConduit
            conduit_console = getattr(conduit, "console", None)
            if conduit_console:
                return conduit_console

        return None

    @console.setter
    def console(self, console: "Console"):
        """Sets the console object for rich output"""
        self.console = console

    @progress_display
    def query(
        self,
        # Standard parameters (match Model.query signature)
        query_input: str | list[Message] | Message | None = None,
        response_model: type["BaseModel"] | None = None,
        cache=True,
        temperature: float | None = None,
        stream: bool = False,
        output_type: OutputType = "text",
        max_tokens: int | None = None,
        # For progress reporting decorator
        verbose: Verbosity = Verbosity.PROGRESS,
        index: int = 0,
        total: int = 0,
        # If we're hand-constructing Request params
        request: Request | None = None,
        # Options for debugging
        return_request: bool = False,
        return_error: bool = False,
    ) -> ConduitResult:
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
                from conduit.tests.fixtures import sample_error

                return sample_error

            # Server doesn't support streaming
            if stream:
                logger.warning(
                    "Server does not support streaming, ignoring stream=True"
                )

            # Execute the query via server
            logger.info("Sending query to Conduit server")
            start_time = time()

            # Use SiphonClient's query_sync method
            response = self._client.conduit.query_sync(request)

            stop_time = time()
            logger.info(
                f"Server query completed in {stop_time - start_time:.2f} seconds"
            )

            # The server should return a Response object
            from conduit.result.response import Response

            if isinstance(response, Response):
                return response
            else:
                raise TypeError(f"Expected Response from server, got {type(response)}")

        except ValidationError as e:
            conduit_error = ConduitError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Validation error: {conduit_error}")
            return conduit_error
        except Exception as e:
            conduit_error = ConduitError.from_exception(
                e,
                code="server_query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Error during server query: {conduit_error}")
            return conduit_error

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

    def __repr__(self) -> str:
        """String representation"""
        return f"ClientModel(model='{self.model}')"
