"""
RemoteModel - A server-based Model implementation that maintains protocol compatibility with Model class.
"""

from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.model.models.modelstore import ModelStore
from conduit.core.model.clients.client import Client, Usage
from conduit.utils.progress.wrappers import progress_display
from conduit.domain.result.result import ConduitResult
from conduit.domain.result.error import ConduitError
from headwater_api.classes import StatusResponse
from pydantic import ValidationError
from typing import override
from time import time
import logging


logger = logging.getLogger(__name__)


class RemoteModel(ModelBase):
    """
    Server-based model client that implements the same protocol as Model class.
    Works seamlessly with Conduit and other components.
    """

    @override
    def get_client(self, model_name: str) -> Client:
        """Get the underlying HeadwaterClient"""
        return ModelStore.get_client(model_name=self.name, execution_mode="remote")

    @progress_display
    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        if kwargs.get("stream", False):
            logger.warning("Server does not support streaming, ignoring stream=True")
        try:
            request = self._prepare_request(query_input, **kwargs)

            if kwargs.get("cache", False):
                cached = self._check_cache(request)
                if cached:
                    return cached

            logger.info("Sending query to Conduit server")
            start_time = time()
            response, _ = self.client.query(request)
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

    @override
    def tokenize(self, text: str) -> int:
        raise NotImplementedError(
            "Tokenization is not supported for RemoteModel. Please use a local model for tokenization."
        )

    # Client/server specific methods
    @property
    def status(self) -> StatusResponse:
        """Get server status"""
        return self.client.get_status()

    @property
    def ping(self) -> bool:
        """Ping server to check health"""
        return self.client.ping()
