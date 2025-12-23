from __future__ import annotations
import asyncio
import logging
from typing import Any, TYPE_CHECKING, override

from conduit.config import settings
from conduit.core.model.model_async import ModelAsync
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import GenerationRequest
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.concurrency.warn import _warn_if_loop_exists

if TYPE_CHECKING:
    from conduit.core.model.modalities.audio import AudioSync
    from conduit.core.model.modalities.image import ImageSync
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.domain.request.query_input import QueryInput
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.message.message import Message
    from rich.console import Console

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

        # Plugins
        self._audio: AudioSync | None = None
        self._image: ImageSync | None = None

    @property
    def audio(self) -> AudioSync:
        """
        Lazy loading of audio generation/analyzation namespace.
        """
        from conduit.core.model.modalities.audio import AudioSync

        if self._audio is None:
            self._audio = AudioSync(parent=self)
        return self._audio

    @property
    def image(self) -> ImageSync:
        """
        Lazy loading of image generation/analyzation namespace.
        """
        from conduit.core.model.modalities.image import ImageSync

        if self._image is None:
            self._image = ImageSync(parent=self)
        return self._image

    def query(
        self, query_input: QueryInput | None = None, **kwargs: Any
    ) -> GenerationResult:
        """
        Synchronous entry point for generation.

        Args:
            query_input: The input messages or string
            **kwargs: Either param overrides (temperature, max_tokens, etc.)
                     or a pre-built 'request' for advanced usage

        Currently accepted kwargs:
            - request: GenerationRequest (bypass normal flow
            - verbosity: Verbosity
            - cache: bool
            - include_history: bool

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
            return self._run_sync(self._impl.pipe(request))

        # Ensure query_input is provided
        if query_input is None:
            raise ValueError("query_input is required")

        # Construct effective params -- these are overrides.
        # First make a copy of stored params, then apply any overrides from kwargs.
        effective_params = self.params.model_copy()
        param_overrides = {
            k: v for k, v in kwargs.items() if k in effective_params.model_fields
        }
        if param_overrides:
            updated_data = effective_params.model_dump()
            updated_data.update(param_overrides)
            effective_params = GenerationParams(**updated_data)
        else:
            effective_params = self.params

        # Build Request from stored params/options
        request = self._impl._prepare_request(
            query_input, effective_params, self.options
        )

        # Implement overrides
        if "verbosity" in kwargs:
            request.verbosity_override = kwargs["verbosity"]
        if "cache" in kwargs:
            request.use_cache = kwargs["cache"]
        if "include_history" in kwargs:
            request.include_history = kwargs["include_history"]

        return self._run_sync(self._impl.query(request))

    def tokenize(self, payload: str | list[Message]) -> int:
        """
        Synchronous entry point for tokenization.
        """
        return self._run_sync(self._impl.tokenize(payload))

    # Factory
    @classmethod
    def create(
        cls,
        model: str,
        *,
        project_name: str = settings.default_project_name,
        persist: bool | str = False,
        cached: bool | str = False,
        verbosity: Verbosity = settings.default_verbosity,
        console: Console | None = None,
        system: str | None = None,
        debug_payload: bool = False,
        **param_kwargs: Any,
    ) -> ModelSync:
        """
        Factory with sensible defaults.

        - `model`: required model name/alias (used for GenerationParams.model).
        - `param_kwargs`: go directly into GenerationParams(...).
        - `cached` / `persist` / `verbosity` / `console`:
            baked into ConduitOptions as baseline behavior.
        """
        # Params: seed with model + any extra params
        params = GenerationParams(model=model, **param_kwargs, system=system)

        # Options: start from global defaults
        options = settings.default_conduit_options()

        # Collect overrides
        opt_updates: dict[str, Any] = {"verbosity": verbosity}

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

        # Apply updates (Pydantic v2)
        options = options.model_copy(update=opt_updates)

        return cls(model=model, params=params, options=options)

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

    @override
    def __repr__(self) -> str:
        return (
            f"ModelSync(model={self._impl.model_name!r}, "
            f"params={self.params!r}, options={self.options!r})"
        )
