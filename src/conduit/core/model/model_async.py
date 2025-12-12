from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.clients.client_base import Client
from conduit.domain.result.result import GenerationResult
from conduit.domain.request.query_input import QueryInput
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from typing import override, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelAsync(ModelBase):
    """
    Async implementation of Model - a stateless "dumb pipe".
    Execution context (params/options) passed explicitly to methods.
    """

    @override
    def get_client(self, model_name: str) -> Client:
        logger.info(f"Retrieving client for model: {model_name}")
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.get_client(model_name, "sdk")

    @override
    async def query(
        self,
        query_input: QueryInput | str | list[Message],
        params: GenerationParams,
        options: ConduitOptions,
    ) -> GenerationResult:
        """
        Execute a query with explicit execution context.

        Args:
            query_input: The input messages or string
            params: Generation parameters (model, temperature, etc.)
            options: Conduit options (cache, console, etc.)

        Returns:
            GenerationResult from the LLM
        """
        logger.info("ModelAsync.query called with model: %s", self.model_name)
        request = self._prepare_request(query_input, params, options)
        result = await self.pipe(request)
        return result

    @override
    async def tokenize(self, payload: str | list[Message]) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        logger.info("ModelAsync.tokenize called with model: %s", self.model_name)
        return await self.client.tokenize(model=self.model_name, payload=payload)


if __name__ == "__main__":
    from conduit.config import settings
    import asyncio

    async def main():
        model = ModelAsync("gpt3")
        params = GenerationParams(model="gpt3")
        options = settings.default_conduit_options()

        result = await model.query("Hello, world!", params, options)
        print(result)

        ts = await model.tokenize("i am the very model of a modern major general")
        print(ts)

    asyncio.run(main())
