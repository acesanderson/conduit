from __future__ import annotations
from conduit.config import settings
from conduit.core.conduit.conduit_async import ConduitAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.utils.concurrency.warn import _warn_if_loop_exists
import asyncio
import logging
from typing import Any, TYPE_CHECKING, override

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


class ConduitBatch:
    """
    Batch execution wrapper for ConduitAsync.

    Supports two modes:
    1. Template mode: input_variables_list + stored Prompt
    2. String mode: prompt_strings_list (pre-rendered prompts)

    Designed for high-throughput batch processing with optional rate limiting.
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
            prompt: Optional prompt template (required for template mode)
            params: LLM parameters (temperature, max_tokens, etc.)
            options: Runtime configuration (caching, console, etc.)
        """
        self.prompt = prompt
        self.params = params or settings.default_params()
        self.options = options or settings.default_conduit_options()
        logger.debug(
            f"ConduitBatch initialized with prompt={self.prompt}, params={self.params}, options={self.options}"
        )

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
        Execute batch processing with concurrent execution.

        Exactly one of input_variables_list or prompt_strings_list must be provided.

        Args:
            input_variables_list:
                List of input variable dicts for template rendering.
                Requires a Prompt to be set on the instance.
            prompt_strings_list:
                List of pre-rendered prompt strings.
                Does not require a Prompt instance.
            max_concurrent:
                Optional maximum number of concurrent requests.
                If None, all requests run concurrently without limit.
                Use this for rate limiting (e.g., max_concurrent=5).
            cached:
                Per-call override for caching.
            persist:
                Per-call override for persistence.
            verbosity:
                Per-call override for progress/logging verbosity.
            param_overrides:
                Dict of fields to merge into GenerationParams.

        Returns:
            list[Conversation]: Completed conversations for each input.

        Raises:
            ValueError: If neither or both input modes are provided.
            ValueError: If input_variables_list is provided without a Prompt.
        """
        logger.info("Starting batch run.")
        logger.debug(f"Input variables list: {input_variables_list}")
        # 1. Validate input mode
        if input_variables_list and prompt_strings_list:
            raise ValueError(
                "Provide exactly one of: input_variables_list OR prompt_strings_list"
            )
        if not input_variables_list and not prompt_strings_list:
            raise ValueError(
                "Must provide either input_variables_list or prompt_strings_list"
            )
        if input_variables_list and not self.prompt:
            raise ValueError(
                "input_variables_list mode requires a Prompt to be set on the instance"
            )

        # 2. Build effective params/options
        effective_params = self._build_params(param_overrides)
        effective_options = self._build_options(
            cached=cached,
            persist=persist,
            verbosity=verbosity,
        )

        # 3. Delegate to async implementation, wrap in sync
        return self._run_sync(
            self._run_batch_async(
                input_variables_list=input_variables_list,
                prompt_strings_list=prompt_strings_list,
                params=effective_params,
                options=effective_options,
                max_concurrent=max_concurrent,
            )
        )

    async def _run_batch_async(
        self,
        input_variables_list: list[dict[str, Any]] | None,
        prompt_strings_list: list[str] | None,
        params: GenerationParams,
        options: ConduitOptions,
        max_concurrent: int | None,
    ) -> list[Conversation]:
        """
        Internal async implementation for batch execution.
        """
        logger.info("Running batch asynchronously.")
        logger.debug(f"Params: {params}, Options: {options}")
        # Create semaphore if max_concurrent is specified
        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        # Create tasks based on mode
        if input_variables_list:
            # Mode 1: Template mode - reuse one ConduitAsync with stored prompt
            conduit = ConduitAsync(self.prompt)
            tasks = [
                self._maybe_with_semaphore(
                    conduit.run(input_vars, params, options),
                    semaphore,
                )
                for input_vars in input_variables_list
            ]
            logger.info(
                f"Executing {len(tasks)} conversations in template mode "
                f"with max_concurrent={max_concurrent or 'unlimited'}"
            )
        else:
            # Mode 2: String mode - create temporary ConduitAsync for each string
            tasks = [
                self._maybe_with_semaphore(
                    ConduitAsync(Prompt(prompt_str)).run(None, params, options),
                    semaphore,
                )
                for prompt_str in prompt_strings_list
            ]
            logger.info(
                f"Executing {len(tasks)} conversations in string mode "
                f"with max_concurrent={max_concurrent or 'unlimited'}"
            )

        # Execute all tasks concurrently
        conversations = await asyncio.gather(*tasks, return_exceptions=False)
        return conversations

    async def _maybe_with_semaphore(
        self,
        coroutine: Any,
        semaphore: asyncio.Semaphore | None,
    ) -> Conversation:
        """
        Optionally wrap a coroutine with semaphore-based rate limiting.
        """
        if semaphore:
            async with semaphore:
                return await coroutine
        else:
            return await coroutine

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
        **param_kwargs: Any,
    ) -> "ConduitBatch":
        """
        Factory with sensible defaults.

        Args:
            model: Required model name/alias (used for GenerationParams.model).
            prompt: Optional str or Prompt (None for string mode).
            project_name: Project name for cache/repository naming.
            persist: Enable persistence (bool or str for custom name).
            cached: Enable caching (bool or str for custom name).
            verbosity: Default verbosity level.
            console: Optional rich Console instance.
            **param_kwargs: Additional parameters merged into GenerationParams.

        Returns:
            ConduitBatch: Configured batch instance.
        """
        # Prompt coercion (optional)
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

        # Collect overrides
        if verbosity is not None:
            opt_updates["verbosity"] = verbosity

        if console is not None:
            opt_updates["console"] = console

        # Cache wiring
        if cached:
            cache_name = cached if isinstance(cached, str) else project_name
            opt_updates["cache"] = settings.default_cache(name=cache_name)

        # Persistence wiring
        if persist:
            repo_name = persist if isinstance(persist, str) else project_name
            opt_updates["repository"] = settings.default_repository(name=repo_name)

        # Apply updates (Pydantic v2)
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
        # Options: start from global defaults
        options = settings.default_conduit_options()

        opt_updates: dict[str, Any] = {}

        # Collect overrides
        if verbosity is not None:
            opt_updates["verbosity"] = verbosity

        # Cache wiring
        if cached:
            cache_name = cached if isinstance(cached, str) else project_name
            opt_updates["cache"] = settings.default_cache(name=cache_name)

        # Persistence wiring
        if persist:
            repo_name = persist if isinstance(persist, str) else project_name
            opt_updates["repository"] = settings.default_repository(name=repo_name)

        # Apply updates (Pydantic v2)
        opts = options.model_copy(update=opt_updates)

        return opts

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
            f"ConduitBatch(prompt={self.prompt!r}, "
            f"params={self.params!r}, options={self.options!r})"
        )
