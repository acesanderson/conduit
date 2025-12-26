"""
ModelRemote - Synchronous, UX-focused wrapper for remote model execution.

This class provides a convenient interface for interacting with models hosted
on a remote server, with the same fat UX as ModelSync.
"""

from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING, override

from conduit.config import settings
from conduit.core.clients.client_base import Client
from conduit.core.model.models.modelstore import ModelStore
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.query_input import QueryInput, constrain_query_input
from conduit.domain.request.request import GenerationRequest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.message.message import Message
    from headwater_api.classes import StatusResponse

logger = logging.getLogger(__name__)


class ModelRemote:
    """
    Synchronous, UX-focused wrapper for remote model execution.

    - Holds *how* to execute: GenerationParams, ConduitOptions.
    - Takes *what* to process at call-time via query_input.
    - Designed for scripts and REPL usage with remote model serving.
    - Includes server-specific features: status, ping, batch operations.
    """

    def __init__(
        self,
        model: str,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the synchronous remote model wrapper.

        Args:
            model: Model name/alias (e.g., "gpt-4o", "claude-sonnet-4").
            params: LLM parameters (temperature, max_tokens, etc.).
                    If not provided, defaults will be used with the specified model.
            options: Runtime configuration (caching, console, etc.).
            **kwargs: Additional parameters merged into GenerationParams (e.g., temperature=0.7).
        """
        from conduit.core.model.models.modelstore import ModelStore

        # 1. Store model identity
        self.model_name: str = ModelStore.validate_model(model)

        # 2. Get remote client (already synchronous)
        self.client: Client = ModelStore.get_client(
            model_name=self.model_name, execution_mode="remote"
        )
        logger.info(f"Initialized remote client for model: {self.model_name}")

        # 3. Store execution context
        if params is None:
            # Create minimal params with correct model
            self.params = GenerationParams(model=self.model_name)
        else:
            # Use provided params, ensuring model matches
            if params.model != self.model_name:
                # Override for consistency
                self.params = params.model_copy(update={"model": self.model_name})
            else:
                self.params = params

        self.options: ConduitOptions = options or settings.default_conduit_options()

        # 4. Merge kwargs into params for convenience
        if kwargs:
            updated_data = self.params.model_dump()
            updated_data.update(kwargs)
            self.params = GenerationParams(**updated_data)

    def query(
        self, query_input: QueryInput | None = None, **kwargs: Any
    ) -> GenerationResult:
        """
        Synchronous entry point for generation.

        Args:
            query_input: The input messages or string
            **kwargs: Either param overrides (temperature, max_tokens, etc.)
                     or a pre-built 'request' for advanced usage

        Returns:
            GenerationResult from the remote LLM
        """
        # Handle pre-built request (advanced usage)
        if "request" in kwargs:
            request = kwargs["request"]
            if not isinstance(request, GenerationRequest):
                raise TypeError(
                    f"request must be a GenerationRequest, got {type(request)}"
                )
            # Call client directly with pre-built request
            return self.client.query(request)

        # Normal flow: build effective params and delegate
        if query_input is None:
            raise ValueError("query_input is required")

        # Handle streaming warning
        if kwargs.get("stream", False):
            logger.warning(
                "Remote models do not support streaming. Ignoring stream=True."
            )
            kwargs["stream"] = False

        effective_params = self._build_params(kwargs)
        request = self._prepare_request(query_input, effective_params, self.options)

        logger.info(f"Sending query to remote server for model: {self.model_name}")
        return self.client.query(request)

    def batch(
        self, query_inputs: list[QueryInput | str | Sequence[Message]], **kwargs: Any
    ) -> list[GenerationResult]:
        """
        Synchronous batch generation.

        Args:
            query_inputs: List of input messages or strings
            **kwargs: Param overrides applied to all requests

        Returns:
            List of GenerationResult objects from the remote LLM
        """
        effective_params = self._build_params(kwargs)

        # Build requests for all inputs
        requests = [
            self._prepare_request(query_input, effective_params, self.options)
            for query_input in query_inputs
        ]

        logger.info(
            f"Sending batch of {len(requests)} queries to remote server "
            f"for model: {self.model_name}"
        )
        return self.client.batch(requests)

    def tokenize(self, payload: str | Sequence[Message]) -> int:
        """
        Synchronous entry point for tokenization.

        Args:
            payload: Text string or list of messages to tokenize

        Returns:
            Token count for the payload
        """
        logger.info(
            f"Requesting tokenization from remote server for model: {self.model_name}"
        )
        return self.client.tokenize(model=self.model_name, payload=payload)

    # Server-specific properties
    @property
    def status(self) -> StatusResponse:
        """Get server status."""
        logger.info("Fetching server status")
        return self.client.get_status()

    @property
    def ping(self) -> bool:
        """Ping server to check health."""
        logger.info("Pinging server")
        return self.client.ping()

    # Config methods - mutate stored options
    def enable_cache(self, name: str = settings.default_project_name) -> None:
        """Enable caching with specified name."""
        self.options.cache = settings.default_cache(name)
        logger.info(f"Enabled cache: {name}")

    def enable_console(self) -> None:
        """Enable rich console output."""
        if self.options.console is None:
            from rich.console import Console

            logger.info("Enabling console.")
            self.options.console = Console()

    def disable_cache(self) -> None:
        """Disable caching."""
        if self.options.cache is not None:
            logger.info("Disabling cache.")
            self.options.cache = None

    def disable_console(self) -> None:
        """Disable console output."""
        if self.options.console is not None:
            logger.info("Disabling console.")
            self.options.console = None

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

    def _build_params(self, param_overrides: dict[str, Any] | None) -> GenerationParams:
        """
        Build effective params by merging stored params with overrides.
        """
        if not param_overrides:
            return self.params

        updated_data = self.params.model_dump()
        updated_data.update(param_overrides)
        return GenerationParams(**updated_data)

    # Class methods for global info
    @classmethod
    def models(cls) -> dict[str, list[str]]:
        """
        Returns a dictionary of available models.
        This is useful for introspection and debugging.
        """
        return ModelStore.models()

    @override
    def __repr__(self) -> str:
        return (
            f"ModelRemote(model={self.model_name!r}, "
            f"params={self.params!r}, options={self.options!r})"
        )


if __name__ == "__main__":
    # Example usage
    model = ModelRemote("gpt3", temperature=0.7)
    model.enable_cache("test")

    # Single query
    result = model.query("Hello, world!")
    print(result)

    # Batch queries
    results = model.batch(["What is Python?", "What is Rust?", "What is Go?"])
    for r in results:
        print(r)

    # Tokenization
    token_count = model.tokenize("I am the very model of a modern major general")
    print(f"Token count: {token_count}")

    # Server health
    print(f"Server status: {model.status}")
    print(f"Server ping: {model.ping}")
