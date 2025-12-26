from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING, override

from conduit.config import settings
from conduit.core.conduit.conduit_async import ConduitAsync
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.core.prompt.prompt import Prompt
from conduit.utils.concurrency.warn import _warn_if_loop_exists

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


class ConduitSync:
    """
    Synchronous, UX-focused wrapper for ConduitAsync.

    - Holds *how* to process: Prompt, GenerationParams, ConduitOptions.
    - Takes *what* to process at call-time via input variables.
    - Designed for scripts and REPL usage where managing state and event loops
      is unnecessary overhead.
    """

    def __init__(
        self,
        prompt: Prompt,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
    ):
        # 1. Instantiate the async implementation (dumb pipe - only needs prompt)
        self._impl = ConduitAsync(prompt)

        # 2. Store execution context for UX
        self.params = params or settings.default_params()
        self.options = options or settings.default_conduit_options()

    # Sugar: payload-only
    def __call__(self, **input_variables: Any) -> Conversation:
        """
        Syntactic sugar for `run(input_variables=...)`.

        IMPORTANT: this is *payload-only*. No params/options knobs here;
        those live on `run()`.
        """
        return self.run(input_variables=input_variables)

    # Main entry point
    def run(
        self,
        input_variables: dict[str, Any] | None = None,
        *,
        cached: bool | None = None,
        persist: bool | None = None,
        verbosity: Verbosity | None = None,
        param_overrides: dict[str, Any] | None = None,
    ) -> Conversation:
        """
        Execute the configured Conduit synchronously.

        Args:
            input_variables:
                Template variables for the prompt string.
            cached:
                Per-call override for caching:
                - None: keep instance-level option
                - False: disable cache
                - True: enable cache (using a default cache if none configured)
            persist:
                Per-call override for persistence:
                - None: keep instance-level option
                - False: disable repository
                - True: enable repository (using a default repo if none configured)
            verbosity:
                Per-call override for progress / logging verbosity.
            param_overrides:
                Dict of fields to merge into GenerationParams
                (e.g. {"temperature": 0.9}).

        Returns:
            Conversation: the final conversation after Engine.run.
        """
        # 1) Build effective params/options from stored state + overrides
        effective_params = self._build_params(param_overrides)
        effective_options = self._build_options(
            cached=cached,
            persist=persist,
            verbosity=verbosity,
        )

        # 2) Delegate to async implementation, wrap in sync
        return self._run_sync(
            self._impl.run(
                input_variables=input_variables,
                params=effective_params,
                options=effective_options,
            )
        )

    # Factory
    @classmethod
    def create(
        cls,
        model: str,
        prompt: Prompt | str,
        *,
        project_name: str = settings.default_project_name,
        persist: bool | str = False,
        cached: bool | str = False,
        verbosity: Verbosity = settings.default_verbosity,
        console: Console | None = None,
        system: str | None = None,  # placeholder for future system-message wiring
        debug_payload: bool = False,
        use_remote: bool = False,
        **param_kwargs: Any,
    ) -> ConduitSync:
        """
        Factory with sensible defaults.

        - `model`: required model name/alias (used for GenerationParams.model).
        - `prompt`: str or Prompt.
        - `param_kwargs`: go directly into GenerationParams(...).
        - `cached` / `persist` / `verbosity` / `console`:
            baked into ConduitOptions as baseline behavior.
        """
        # Prompt coercion
        if isinstance(prompt, str):
            prompt_obj = Prompt(prompt)
        elif isinstance(prompt, Prompt):
            prompt_obj = prompt
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        # Params: seed with model + any extra params
        params = GenerationParams(model=model, **param_kwargs, system=system)

        # Options: start from global defaults
        options = settings.default_conduit_options()

        # Collect overrides
        opt_updates = {"verbosity": verbosity}

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

        # Debug
        if debug_payload:
            opt_updates["debug_payload"] = True

        # Remote execution
        if use_remote:
            opt_updates["use_remote"] = True

        # Apply updates (Pydantic v2)
        options = options.model_copy(update=opt_updates)

        return cls(prompt=prompt_obj, params=params, options=options)

    # Helpers
    def _build_params(self, param_overrides: dict[str, Any] | None) -> GenerationParams:
        base = self.params or settings.default_params
        if not param_overrides:
            return base
        return base.model_copy(update=param_overrides)

    def _build_options(
        self,
        *,
        cached: bool | None,
        persist: bool | None,
        verbosity: Verbosity | None,
    ) -> ConduitOptions:
        opts = self.options or settings.default_conduit_options()

        if verbosity is not None:
            opts.verbosity = verbosity

        # Simple enable/disable knobs; names are handled in `create`
        if cached is not None:
            if not cached:
                opts.cache = None
            elif opts.cache is None:
                opts.cache = settings.default_cache(name=settings.default_project_name)

        if persist is not None:
            if not persist:
                opts.repository = None
            elif opts.repository is None:
                opts.repository = settings.default_repository(
                    name=settings.default_project_name
                )

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
        _warn_if_loop_exists()
        try:
            return asyncio.run(coroutine)
        except KeyboardInterrupt:
            logger.warning("Operation cancelled by user.")
            raise

    def __getattr__(self, name: str) -> object:
        """
        Proxy attribute access to the underlying ConduitAsync instance.
        Allows access to properties like self.prompt, etc.
        """
        return getattr(self._impl, name)

    @override
    def __repr__(self) -> str:
        return (
            f"ConduitSync(prompt={self._impl.prompt!r}, "
            f"params={self.params!r}, options={self.options!r})"
        )


if __name__ == "__main__":
    # Simple usage example
    conduit = ConduitSync.create(
        model="gpt3",
        prompt="Hello, {{name}}! How can I assist you today?",
        persist=True,
        cached=True,
        temperature=0.7,
        system="You will always response like an effete aristocrat.",
        debug_payload=True,
    )

    conversation = conduit(name="Alice")
    print(conversation.last.content)
