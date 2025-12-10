from __future__ import annotations
from conduit.config import settings
from conduit.domain.request.request import GenerationRequest
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.query_input import QueryInput, constrain_query_input
from conduit.core.clients.client_base import Client
from conduit.storage.odometer.odometer_registry import OdometerRegistry
from conduit.middleware.middleware import middleware
from typing import TYPE_CHECKING, override
import logging

# Load only if type checking
if TYPE_CHECKING:
    from rich.console import Console
    from conduit.domain.message.message import Message
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.config.conduit_options import ConduitOptions

logger = logging.getLogger(__name__)


class ModelBase:
    """
    Stem class for Model implementations; not to be used directly.
    """

    # Class singleton
    odometer_registry: OdometerRegistry = OdometerRegistry()

    def __init__(
        self,
        options: ConduitOptions | None = None,
        params: GenerationParams | None = None,
    ):
        from conduit.core.model.models.modelstore import ModelStore

        # Initial attributes
        self.options: ConduitOptions = options or settings.default_conduit_options()
        self.params: GenerationParams = params or settings.default_params
        # Coerce model name
        self.model_name: str = ModelStore.validate_model(params.model)
        self.client: Client = self.get_client(model_name=self.model_name)

    # Optional config methods (post-init)
    def enable_cache(self, name: str = settings.default_project_name) -> None:
        self.options.cache = settings.default_cache(name)

    def enable_console(self) -> None:
        if self.options.console is None:
            logger.info("Enabling console.")
            self.options.console = Console()

    def disable_cache(self) -> None:
        if self.options.cache is not None:
            logger.info("Disabling cache.")
            self.options.cache = None

    def disable_console(self) -> None:
        if self.options.console is not None:
            logger.info("Disabling console.")
            self.options.console = None

    # Class methods for global info
    @classmethod
    def models(cls) -> dict[str, list[str]]:
        """
        Returns a dictionary of available models.
        This is useful for introspection and debugging.
        """
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.models()

    @classmethod
    def stats(cls):
        """
        Pretty prints session statistics (from OdometerRegistry.session_odometer).
        """
        cls.odometer_registry.session_odometer.stats()

    # Our pipe method
    @middleware
    async def pipe(self, request: GenerationRequest) -> GenerationResult:
        return await self.client.query(request)

    # Helper methods
    def _prepare_request(
        self, query_input: QueryInput | None = None, **kwargs: object
    ) -> GenerationRequest:
        """
        PURE CPU: Constructs and validates the GenerationRequest object.
        """
        # First, see if we have a request already, if so, pass through
        try:
            request = kwargs["request"]
            if not isinstance(request, GenerationRequest):
                raise TypeError(
                    f"request must be a GenerationRequest, got {type(request)}"
                )
            return request
        except KeyError:
            pass

        # Otherwise, build request from query_input
        if query_input is None:
            raise ValueError("query_input is required when no request is provided.")
        # constrain query_input per model capabilities
        query_input: list[Message] = constrain_query_input(query_input=query_input)
        # construct GenerationParams
        params = GenerationParams(
            model=self.model_name,
            **kwargs,
        )

        kwargs["model"] = self.model_name
        request = GenerationRequest(
            messages=query_input,
            params=params,
        )
        return request

    # Expected methods in subclasses
    def get_client(self, model_name: str) -> Client:
        raise NotImplementedError(
            "get_client must be implemented in subclasses (sync or async)."
        )

    async def query(
        self, query_input: QueryInput, **kwargs: object
    ) -> GenerationResult:
        raise NotImplementedError(
            "query must be implemented in subclasses (sync or async)."
        )

    async def tokenize(self, payload: str | list[Message]) -> int:
        raise NotImplementedError("tokenize must be implemented in subclasses.")

    # Dunders
    @override
    def __repr__(self) -> str:
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
