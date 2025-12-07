"""
MAJOR REFACTOR INCOMING:
- Model is really a collection of factories and orchestrators around Clients.
- It's a "Request" factory.
- All query methods pass through _execute, which is wrapped with middleware for caching, progress, odometer, etc.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.request.request import Request
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.query_input import QueryInput, constrain_query_input
from conduit.core.clients.client_base import Client
from conduit.utils.progress.verbosity import Verbosity
from conduit.storage.odometer.odometer_registry import OdometerRegistry
from conduit.middleware.middleware import middleware_sync, middleware_async
from typing import TYPE_CHECKING, override
import logging

# Load only if type checking
if TYPE_CHECKING:
    from rich.console import Console
    from conduit.storage.cache.protocol import ConduitCache
    from conduit.domain.message.message import Message
    from conduit.domain.request.request import Request
    from conduit.domain.result.result import ConduitResult

logger = logging.getLogger(__name__)


class ModelBase:
    """
    Stem class for Model implementations; not to be used directly.
    """

    # Class singleton
    odometer_registry: OdometerRegistry = OdometerRegistry()

    def __init__(
        self,
        model_name: str = settings.preferred_model,
        console: Console | None = settings.default_console,
        verbosity: Verbosity = settings.default_verbosity,
        cache_engine: str | ConduitCache | None = None,
    ):
        from conduit.core.model.models.modelstore import ModelStore

        self.model_name: str = ModelStore.validate_model(model_name)
        self.verbosity: Verbosity = verbosity
        self.console: Console | None = console
        self.client: Client = self.get_client(model_name=self.model_name)

        if cache_engine is not None:
            from conduit.storage.cache.protocol import ConduitCache

            if isinstance(cache_engine, str):
                # Convenience: Create cache from string name
                logger.info(f"Initializing cache with name: '{cache_engine}'")
                self.cache: str | ConduitCache | None = ConduitCache(name=cache_engine)
            else:  # It's already a ConduitCache instance
                self.cache = cache_engine
        else:
            self.cache = None

    # Optional config methods (post-init)
    def enable_cache(self) -> None:
        if self.cache is None:
            from conduit.storage.cache.cache import ConduitCache

            logger.info("Enabling default cache.")
            self.cache = ConduitCache()

    def enable_console(self) -> None:
        if self.console is None:
            logger.info("Enabling console.")
            self.console = Console()

    def disable_cache(self) -> None:
        if self.cache is not None:
            logger.info("Disabling cache.")
            self.cache = None

    def disable_console(self) -> None:
        if self.console is not None:
            logger.info("Disabling console.")
            self.console = None

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

    # Query methods: these are orchestrated in subclasses
    @middleware_sync
    def _execute(self, request: Request) -> ConduitResult:
        return self.client.query(request)

    @middleware_async
    async def _execute_async(self, request: Request) -> ConduitResult:
        return await self.query.query_async(request)

    def _prepare_request(
        self, query_input: QueryInput | None = None, **kwargs: object
    ) -> Request:
        """
        PURE CPU: Constructs and validates the Request object.
        """
        # First, see if we have a request already, if so, pass through
        try:
            request = kwargs["request"]
            if not isinstance(request, Request):
                raise TypeError(f"request must be a Request, got {type(request)}")
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
        request = Request(
            messages=query_input,
            params=params,
        )
        return request

    # Expected methods in subclasses
    def get_client(self, model_name: str) -> Client:
        raise NotImplementedError(
            "get_client must be implemented in subclasses (sync or async)."
        )

    def query(self, query_input: QueryInput, **kwargs: object) -> ConduitResult:
        raise NotImplementedError(
            "query must be implemented in subclasses (sync or async)."
        )

    def tokenize(self, text: str) -> int:
        raise NotImplementedError("tokenize must be implemented in subclasses.")

    # Dunders
    @override
    def __repr__(self) -> str:
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
