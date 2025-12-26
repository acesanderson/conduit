from __future__ import annotations
from conduit.domain.request.request import GenerationRequest
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.query_input import QueryInput, constrain_query_input
from conduit.core.clients.client_base import Client
from conduit.middleware.middleware import middleware
from typing import TYPE_CHECKING, override
import logging

# Load only if type checking
if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.domain.message.message import Message
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.core.model.modalities.audio import AudioSync, AudioAsync
    from conduit.core.model.modalities.image import ImageSync, ImageAsync

logger = logging.getLogger(__name__)


class ModelBase:
    """
    Stem class for Model implementations; not to be used directly.
    Holds only model identity (model_name + client).
    Execution context (params/options) passed to methods, not stored.
    """

    def __init__(self, model: str):
        """
        Initialize the Model base with only its identity.

        Args:
            model: The model name/alias (e.g., "gpt-4o", "claude-sonnet-4")
        """
        from conduit.core.model.models.modelstore import ModelStore

        # Model identity - the only thing stored
        self.model_name: str = ModelStore.validate_model(model)
        self.client: Client = self.get_client(model_name=self.model_name)
        # Plugins
        self._audio: AudioSync | AudioAsync | None = None
        self._image: ImageSync | ImageAsync | None = None

    # Class methods for global info
    @classmethod
    def models(cls) -> dict[str, list[str]]:
        """
        Returns a dictionary of available models.
        This is useful for introspection and debugging.
        """
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.models()

    # Our pipe method
    @middleware
    async def pipe(self, request: GenerationRequest) -> GenerationResult:
        """
        Core delegation point - passes request to client.
        Options used by middleware for caching, console, etc.
        """
        return await self.client.query(request)

    # Helper methods
    def _prepare_request(
        self, query_input: QueryInput, params: GenerationParams, options: ConduitOptions
    ) -> GenerationRequest:
        """
        PURE CPU: Constructs and validates the GenerationRequest object.
        """
        # Constrain query_input per model capabilities
        query_input_list: Sequence[Message] = constrain_query_input(
            query_input=query_input
        )

        # Ensure params has correct model name
        if params.model != self.model_name:
            params = params.model_copy(update={"model": self.model_name})

        request = GenerationRequest(
            messages=query_input_list,
            params=params,
            options=options,
        )
        return request

    # Abstract methods (must be implemented by subclasses)
    def get_client(self, model_name: str) -> Client:
        raise NotImplementedError("get_client must be implemented in subclasses.")

    async def query(
        self,
        query_input: QueryInput,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> GenerationResult:
        raise NotImplementedError("query must be implemented in subclasses.")

    async def tokenize(self, payload: str | Sequence[Message]) -> int:
        raise NotImplementedError("tokenize must be implemented in subclasses.")

    # Dunders
    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"
