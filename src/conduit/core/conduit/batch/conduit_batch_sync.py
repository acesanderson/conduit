from __future__ import annotations
import asyncio
import logging
from typing import Any, TYPE_CHECKING, override

from conduit.config import settings
from conduit.core.conduit.batch.conduit_batch_async import ConduitBatchAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.utils.concurrency.warn import _warn_if_loop_exists

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


class ConduitBatchSync:
    """
    Synchronous, UX-focused wrapper for ConduitBatchAsync.

    - Holds *how* to process: Prompt, GenerationParams, ConduitOptions.
    - Takes *what* to process at call-time via inputs lists.
    - Manages state, configuration, and the event loop entry point.
    """

    def __init__(
        self,
        prompt: Prompt | None = None,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
    ):
        """
        Initialize the batch conduit.

        Args:
            prompt: Optional prompt template.
            params: Default LLM parameters.
            options: Default Runtime configuration.
        """
        # 1. Instantiate the async implementation
        self._impl = ConduitBatchAsync(prompt)

        # 2. Store execution context for UX
        self.params = params or settings.default_params()
        self.options = options or settings.default_conduit_options()

        logger.debug(
            f"ConduitBatchSync initialized with prompt={self._impl.prompt}, "
            f"params={self.params}, options={self.options}"
        )

    # Main entry point
    def run(
        self,
        input_variables_list: list[dict[str, Any]] | None = None,
        prompt_strings_list: list[str] | None = None,
        *,
        max_concurrent: int | None = None,
        cached: bool | None = None,
        persist: bool | None = None,
        verbosity: Verbosity | None = None,
        param_overrides: dict[str, Any] | None = None,
    ) -> list[Conversation]:
        """
        Execute batch processing synchronously.

        Args:
            input_variables_list: List of input variable dicts (requires Prompt).
            prompt_strings_list: List of pre-rendered prompt strings.
            max_concurrent: Max concurrent requests (for rate limiting).
            cached: Per-call override for caching.
            persist: Per-call override for persistence.
            verbosity: Per-call override for logging.
            param_overrides: Dict merged into GenerationParams.

        Returns:
            list[Conversation]: Completed conversations.
        """
        # 1. Build effective params/options from stored state + overrides
        effective_params = self._build_params(param_overrides)
        effective_options = self._build_options(
            cached=cached,
            persist=persist,
            verbosity=verbosity,
        )

        # 2. Delegate to async implementation, wrap in sync
        return self._run_sync(
            self._impl.run(
                input_variables_list=input_variables_list,
                prompt_strings_list=prompt_strings_list,
                params=effective_params,
                options=effective_options,
                max_concurrent=max_concurrent,
            )
        )

    # Factory
    @classmethod
    def create(
        cls,
        model: str,
        prompt: Prompt | str | None = None,
        *,
        project_name: str = settings.default_project_name,
        persist: bool | str = False,
        cached: bool | str = False,
        verbosity: Verbosity = settings.default_verbosity,
        console: Console | None = None,
        use_remote: bool = False,
        **param_kwargs: Any,
    ) -> "ConduitBatchSync":
        """
        Factory with sensible defaults.
        """
        # Prompt coercion
        prompt_obj = None
        if prompt is not None:
            if isinstance(prompt, str):
                prompt_obj = Prompt(prompt)
            elif isinstance(prompt, Prompt):
                prompt_obj = prompt
            else:
                raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        # Params: seed with model + any extra params
        params = GenerationParams(model=model, **param_kwargs)

        # Options: start from global defaults
        options = settings.default_conduit_options(name=project_name)

        opt_updates: dict[str, Any] = {}

        if verbosity is not None:
            opt_updates["verbosity"] = verbosity

        if console is not None:
            opt_updates["console"] = console

        if cached:
            cache_name = cached if isinstance(cached, str) else project_name
            opt_updates["cache"] = settings.default_cache(name=cache_name)

        if persist:
            repo_name = persist if isinstance(persist, str) else project_name
            opt_updates["repository"] = settings.default_repository(name=repo_name)

        if use_remote:
            opt_updates["use_remote"] = True

        options = options.model_copy(update=opt_updates)

        return cls(prompt=prompt_obj, params=params, options=options)

    # Helpers
    def _build_params(self, param_overrides: dict[str, Any] | None) -> GenerationParams:
        """Build effective params by merging stored params with overrides."""
        if not param_overrides:
            return self.params

        updated_data = self.params.model_dump()
        updated_data.update(param_overrides)
        return GenerationParams(**updated_data)

    def _build_options(
        self,
        *,
        cached: bool | None,
        persist: bool | None,
        verbosity: Verbosity | None,
    ) -> ConduitOptions:
        """Build effective options by merging stored options with overrides."""
        # Start with stored options
        options = self.options or settings.default_conduit_options()
        opt_updates: dict[str, Any] = {}

        if verbosity is not None:
            opt_updates["verbosity"] = verbosity

        # Handle bool flags vs existing config objects
        if cached is not None:
            if not cached:
                opt_updates["cache"] = None
            elif options.cache is None:
                # Enable default cache if requested but not present
                opt_updates["cache"] = settings.default_cache(
                    name=settings.default_project_name
                )

        if persist is not None:
            if not persist:
                opt_updates["repository"] = None
            elif options.repository is None:
                # Enable default repo if requested but not present
                opt_updates["repository"] = settings.default_repository(
                    name=settings.default_project_name
                )

        # Apply updates non-destructively
        return options.model_copy(update=opt_updates)

    # Config methods - mutate stored options
    def enable_cache(self, name: str = settings.default_project_name) -> None:
        """Enable caching with specified name."""
        self.options.cache = settings.default_cache(name)
        logger.info(f"Enabled cache: {name}")

    def enable_repository(self, name: str = settings.default_project_name) -> None:
        """Enable persistence with specified name."""
        self.options.repository = settings.default_repository(name)
        logger.info(f"Enabled repository: {name}")

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

    def disable_repository(self) -> None:
        """Disable persistence."""
        if self.options.repository is not None:
            logger.info("Disabling repository.")
            self.options.repository = None

    def disable_console(self) -> None:
        """Disable console output."""
        if self.options.console is not None:
            logger.info("Disabling console.")
            self.options.console = None

    # Async plumbing
    def _run_sync(self, coroutine: Any) -> Any:
        """Helper to run async methods synchronously."""
        _warn_if_loop_exists()
        try:
            return asyncio.run(coroutine)
        except KeyboardInterrupt:
            logger.warning("Batch operation cancelled by user.")
            raise

    @override
    def __repr__(self) -> str:
        return (
            f"ConduitBatchSync(prompt={self._impl.prompt!r}, "
            f"params={self.params!r}, options={self.options!r})"
        )
