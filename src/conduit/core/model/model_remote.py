"""
RemoteModel - A server-based Model implementation that maintains protocol compatibility with Model class.
"""

from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.model.models.modelstore import ModelStore
from conduit.core.clients.client_base import Client
from conduit.domain.result.result import ConduitResult
from headwater_api.classes import StatusResponse
from typing import override, TYPE_CHECKING
from time import time
import logging

if TYPE_CHECKING:
    from conduit.domain.message.message import Message

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

    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        if kwargs.get("stream", False):
            logger.warning("Server does not support streaming, ignoring stream=True")
        request = self._prepare_request(query_input, **kwargs)

        if kwargs.get("cache", False):
            cached = self._check_cache(request)
            if cached:
                return cached

        logger.info("Sending query to Conduit server")
        start_time = time()
        response = self.client.query(request)
        stop_time = time()
        logger.info(f"Server query completed in {stop_time - start_time:.2f} seconds")

    @override
    def tokenize(self, payload: str | list[Message]) -> int:
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
