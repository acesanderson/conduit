from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.clients.client_base import Client
from conduit.core.model.models.modelstore import ModelStore
from conduit.domain.result.result import ConduitResult
from conduit.domain.result.error import ConduitError
from typing import override, TYPE_CHECKING
import asyncio
import logging

if TYPE_CHECKING:
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelAsync(ModelBase):
    @override
    def get_client(self, model_name: str) -> Client:
        return ModelStore.get_client(model_name, "async")

    @override
    async def query_async(self, query_input=None, **kwargs) -> ConduitResult:
        try:
            request = self._prepare_request(query_input, **kwargs)
            conduit_result = await self._execute_async(request, **kwargs)
            return conduit_result
        except Exception as e:
            try:
                request_request = request.model_dump()
            except Exception:
                request_request = {}
            conduit_error = ConduitError.from_exception(
                e,
                code="query_error",
                category="client",
                request_request=request_request,
            )
            logger.error(f"Error during query: {conduit_error}")
            return conduit_error

    @override
    async def tokenize_async(self, payload: str | list[Message]) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return await asyncio.to_thread(
            self.client.tokenize, model=self.model_name, payload=payload
        )


if __name__ == "__main__":
    import asyncio

    async def main():
        model = ModelAsync(model_name="gpt3")
        result = await model.query_async(query_input="Hello, world!")
        print(result)

    asyncio.run(main())
