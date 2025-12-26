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
    from collections.abc import Sequence
    from conduit.domain.request.request import GenerationRequest
    from conduit.core.model.modalities.audio import AudioAsync
    from conduit.core.model.modalities.image import ImageAsync
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelAsync(ModelBase):
    """
    Async implementation of Model - a stateless "dumb pipe".
    Execution context (params/options) passed explicitly to methods.
    """

    def __init__(self, model: str):
        """
        Initialize the async model with only its identity.

        Args:
            model: The model name/alias (e.g., "gpt-4o", "claude-sonnet-4")
        """
        super().__init__(model=model)
        # Plugins
        self._audio: AudioAsync | None = None
        self._image: ImageAsync | None = None

    @property
    def audio(self) -> AudioAsync:
        """
        Lazy loading of audio generation/analyzation namespace.
        """
        from conduit.core.model.modalities.audio import AudioAsync

        if self._audio is None:
            self._audio = AudioAsync(parent=self)
        return self._audio

    @property
    def image(self) -> ImageAsync:
        """
        Lazy loading of image generation/analyzation namespace.
        """
        from conduit.core.model.modalities.image import ImageAsync

        if self._image is None:
            self._image = ImageAsync(parent=self)
        return self._image

    @override
    def get_client(self, model_name: str) -> Client:
        logger.info(f"Retrieving client for model: {model_name}")
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.get_client(model_name, "sdk")

    @override
    async def query(
        self,
        request: GenerationRequest | None = None,
        query_input: QueryInput | str | Sequence[Message] | None = None,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
    ) -> GenerationResult:
        """
        Execute a query with explicit execution context.
        You have two options:
        - Provide a pre-built GenerationRequest via 'request' (other args ignored)
        - Provide 'query_input', 'params', and 'options' (all required)

        Args:
            request: Pre-built GenerationRequest (optional)
            query_input: The input messages or string
            params: Generation parameters (model, temperature, etc.)
            options: Conduit options (cache, console, etc.)

        Returns:
            GenerationResult from the LLM
        """
        # First, pass through the request if provided
        if request is not None:
            logger.info("ModelAsync.query called with pre-built request")
            result = await self.pipe(request)
            return result
        # Otherwise, the other three args are mandatory
        else:
            if query_input is None or params is None or options is None:
                raise ValueError(
                    "If 'request' is not provided, 'query_input', 'params', and 'options' must all be provided."
                )
            logger.info("ModelAsync.query called with model: %s", self.model_name)
            request = self._prepare_request(query_input, params, options)
            result = await self.pipe(request)
            return result

    @override
    async def tokenize(self, payload: str | Sequence[Message]) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        logger.info("ModelAsync.tokenize called with model: %s", self.model_name)
        return await self.client.tokenize(model=self.model_name, payload=payload)
