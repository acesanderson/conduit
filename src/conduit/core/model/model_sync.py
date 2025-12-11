from __future__ import annotations
import asyncio
import logging
from typing import Any, TYPE_CHECKING

from conduit.config import settings
from conduit.core.model.model_async import ModelAsync
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import GenerationRequest
from conduit.utils.concurrency.warn import _warn_if_loop_exists

if TYPE_CHECKING:
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.domain.request.query_input import QueryInput
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelSync:
    """
    Synchronous, UX-focused wrapper for ModelAsync.

    - Holds *how* to execute: GenerationParams, ConduitOptions.
    - Takes *what* to process at call-time via query_input.
    - Designed for scripts and REPL usage where managing state and event loops
      is unnecessary overhead.
    """

    def __init__(
        self,
        model: str,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the synchronous model wrapper.

        Args:
            model: Model name/alias (e.g., "gpt-4o", "claude-sonnet-4").
            params: LLM parameters (temperature, max_tokens, etc.).
                    If not provided, defaults will be used with the specified model.
            options: Runtime configuration (caching, console, etc.).
            **kwargs: Additional parameters merged into GenerationParams (e.g., temperature=0.7).
        """
        # 1. Instantiate the async implementation (dumb pipe - only needs model identity)
        self._impl = ModelAsync(model)

        # 2. Store execution context
        if params is None:
            # Create minimal params with correct model
            self.params = GenerationParams(model=model)
        else:
            # Use provided params, ensuring model matches
            if params.model != model:
                # Override for consistency
                self.params = params.model_copy(update={"model": model})
            else:
                self.params = params

        self.options: ConduitOptions = options or settings.default_conduit_options()

        # 3. Merge kwargs into params for convenience
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
            GenerationResult from the LLM
        """
        # Handle pre-built request (advanced usage)
        if "request" in kwargs:
            request = kwargs["request"]
            if not isinstance(request, GenerationRequest):
                raise TypeError(
                    f"request must be a GenerationRequest, got {type(request)}"
                )
            # Bypass normal flow and call pipe directly
            return self._run_sync(self._impl.pipe(request, self.options))

        # Normal flow: build effective params and delegate
        if query_input is None:
            raise ValueError("query_input is required")

        effective_params = self._build_params(kwargs)
        return self._run_sync(
            self._impl.query(query_input, effective_params, self.options)
        )

    def tokenize(self, payload: str | list[Message]) -> int:
        """
        Synchronous entry point for tokenization.
        """
        return self._run_sync(self._impl.tokenize(payload))

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
    def _build_params(self, param_overrides: dict[str, Any] | None) -> GenerationParams:
        """
        Build effective params by merging stored params with overrides.
        """
        if not param_overrides:
            return self.params

        updated_data = self.params.model_dump()
        updated_data.update(param_overrides)
        return GenerationParams(**updated_data)

    def _run_sync(self, coroutine: Any) -> Any:
        """
        Helper to run async methods synchronously.
        """
        _warn_if_loop_exists()
        try:
            return asyncio.run(coroutine)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully during blocking calls
            logger.warning("Operation cancelled by user.")
            raise

    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to the underlying ModelAsync instance.
        Allows access to properties like self.model_name, self.client, etc.
        """
        return getattr(self._impl, name)

    def __repr__(self) -> str:
        return (
            f"ModelSync(model={self._impl.model_name!r}, "
            f"params={self.params!r}, options={self.options!r})"
        )


if __name__ == "__main__":
    model = ModelSync("gpt3", temperature=1.0)
    model.enable_cache("test")
    result = model.query("Hello, world!")
    print(result)
