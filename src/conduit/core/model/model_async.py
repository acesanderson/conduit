from conduit.core.model.model_base import ModelBase
from conduit.core.clients.client_base import Client
from conduit.core.model.models.modelstore import ModelStore
from conduit.domain.result.result import ConduitResult
from conduit.domain.result.error import ConduitError
from time import time
from typing import override
from pydantic import ValidationError
import asyncio
import logging

logger = logging.getLogger(__name__)


class ModelAsync(ModelBase):
    @override
    def get_client(self, model_name: str) -> Client:
        return ModelStore.get_client(model_name, "async")

    @override
    async def query(self, query_input=None, **kwargs) -> ConduitResult:
        try:
            request = self._prepare_request(query_input, **kwargs)
            conduit_result = await self._execute_async(request, **kwargs)
            return conduit_result
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
    async def tokenize(self, text: str) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return await asyncio.to_thread(self.client.tokenize, model=self.name, text=text)
