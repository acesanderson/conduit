from __future__ import annotations
import asyncio
import logging
from typing import Any, TYPE_CHECKING

from conduit.core.model.model_async import ModelAsync
from conduit.domain.request.generation_params import GenerationParams
from conduit.utils.concurrency.warn import _warn_if_loop_exists

if TYPE_CHECKING:
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.domain.request.query_input import QueryInput
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelSync:
    """
    Synchronous wrapper for ModelAsync.
    Designed for scripts and REPL usage where managing an event loop is unnecessary overhead.
    """

    def __init__(
        self,
        model: str | None = None,
        options: ConduitOptions | None = None,
        params: GenerationParams | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the synchronous model wrapper.

        Args:
            model: Easy-access string alias (e.g., "gpt-4o"). Overrides params.model if provided.
            options: Runtime configuration (logging, caching).
            params: LLM parameters.
            **kwargs: Additional parameters merged into GenerationParams (e.g., temperature=0.7).
        """
        # 1. UX Improvement: Allow string initialization
        if model:
            if params is None:
                # Create minimal params if none exist
                params = GenerationParams(model=model)
            else:
                # Override model in existing params
                params.model = model

        # 2. Merge kwargs into params for convenience
        if kwargs and params:
            # Update pydantic model with kwargs
            updated_data = params.model_dump()
            updated_data.update(kwargs)
            params = GenerationParams(**updated_data)
        elif kwargs and params is None:
            raise ValueError("Must provide 'model' or 'params' to initialize Model.")

        # 3. Instantiate the heavy lifter
        self._impl = ModelAsync(options=options, params=params)

    def query(
        self, query_input: QueryInput | None = None, **kwargs: Any
    ) -> GenerationResult:
        """
        Synchronous entry point for generation.
        Wraps asyncio.run() around the async implementation.
        """
        return self._run_sync(self._impl.query(query_input, **kwargs))

    def tokenize(self, payload: str | list[Message]) -> int:
        """
        Synchronous entry point for tokenization.
        """
        return self._run_sync(self._impl.tokenize(payload))

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
        return f"ModelSync(wrapping={self._impl!r})"


if __name__ == "__main__":
    model = ModelSync("gpt3", temperature=1.0)
    model.enable_cache("test")
    result = model.query("Hello, world!")
    print(result)
