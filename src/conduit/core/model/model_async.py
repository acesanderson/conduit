from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.clients.client_base import Client
from conduit.domain.result.result import GenerationResult
from conduit.domain.request.query_input import QueryInput
from typing import override, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelAsync(ModelBase):
    @override
    def get_client(self, model_name: str) -> Client:
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.get_client(model_name, "sdk")

    @override
    async def query(
        self, query_input: QueryInput | None = None, **kwargs
    ) -> GenerationResult:
        request = self._prepare_request(query_input, **kwargs)
        conduit_result = await self.pipe(request)
        return conduit_result

    @override
    async def tokenize(self, payload: str | list[Message]) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return await self.client.tokenize(model=self.model_name, payload=payload)


if __name__ == "__main__":
    from conduit.domain.request.generation_params import GenerationParams
    import asyncio

    async def main():
        model = ModelAsync(params=GenerationParams(model="gpt3"))
        result = await model.query("Hello, world!")
        print(result)
        ts = await model.tokenize("i am the very model of a modern major general")
        print(ts)

    asyncio.run(main())
