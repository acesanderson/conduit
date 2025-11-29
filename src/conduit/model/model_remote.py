"""
RemoteModel - A server-based Model implementation that maintains protocol compatibility with Model class.
"""
from __future__ import annotations
from headwater_client.client.headwater_client import HeadwaterClient
from conduit.model.models.modelstore import ModelStore
from conduit.progress.wrappers import progress_display
from conduit.result.result import ConduitResult
from conduit.result.error import ConduitError
from pydantic import ValidationError
from time import time
import logging


logger = logging.getLogger(__name__)


class RemoteModel:
    """
    Server-based model client that implements the same protocol as Model class.
    Works seamlessly with Conduit and other components.
    """

    def __init__(self, model: str = "gpt-oss:latest", console: Console | None = None):
        """
        Initialize RemoteModel with server connection.

        Args:
            model: Model name (must be available on server)
            console: Optional Rich console for progress display
        """
        self.model: str = model
        self._console: Console = console
        self._client: HeadwaterClient = self.get_client()
        self._validate_server_model()

    def get_client(self) -> Client:
        """Get the underlying HeadwaterClient"""
        return ModelStore.get_client(model_name=self.model, execution_mode = "remote")


    @property
    def status(self) -> StatusResponse:
        """Get server status"""
        return self._client.get_status()

    @property
    def ping(self) -> bool:
        """Ping server to check health"""
        return self._client.ping()

    @progress_display
    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        if kwargs.get("stream", False):
            logger.warning(
                "Server does not support streaming, ignoring stream=True"
            )
        try:
            # 1. CPU: Prepare
            request = self._prepare_request(query_input, **kwargs)

            # 2. I/O: Cache Read (Blocking)
            if kwargs.get("cache", False):
                cached = self._check_cache(request)
                if cached:
                    return cached

            # 3. I/O: Network Call (Blocking)
            start = time()
            raw_result, usage = self._client.query(request)
            stop = time()

            # 4. CPU: Process
            logger.info("Sending query to Conduit server")
            start_time = time()

            # Use SiphonClient's query_sync method
            response = self._client.conduit.query_sync(request)

            stop_time = time()
            logger.info(
                f"Server query completed in {stop_time - start_time:.2f} seconds"
            )

            # 5. I/O: Cache Write (Blocking)
            if kwargs.get("cache", False):
                self._save_cache(request, response)

            return response

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
                code="query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Error during query: {conduit_error}")
            return conduit_error


@progress_display
    def query() -> ConduitResult:
        """
        Query the server model - matches Model.query() signature exactly
        """

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

    @override
    def tokenize(self, text: str) -> int: ...
