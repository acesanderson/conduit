from __future__ import annotations
import asyncio
import logging
from typing import Any, TYPE_CHECKING, override
from dataclasses import fields, replace
from conduit.core.model.model_async import ModelAsync
from conduit.domain.request.generation_params import GenerationParams
from conduit.core.prompt.prompt import Prompt
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
    A Configured Pipe -- the UX sineater of the Conduit framework.
    Designed for scripts and REPL usage where managing an event loop is unnecessary overhead.
    State regarding 'How to process' is held in self (Prompt, Params, Options).
    State regarding 'What to process' is passed to run().
    """

    def __init__(
        self,
        # Required
        prompt: Prompt,
        # Optional -- a "stash" of default configurations
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
    ):
        """
        Initialize the synchronous model wrapper.

        Args:
            model: Easy-access string alias (e.g., "gpt-4o"). Overrides params.model if provided.
            options: Runtime configuration (logging, caching).
            params: LLM parameters.
            **kwargs: Additional parameters merged into GenerationParams (e.g., temperature=0.7).
        """
        self.prompt: Prompt = prompt
        self.params: GenerationParams | None = params
        self.options: ConduitOptions | None = options

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

    @override
    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to the underlying ModelAsync instance.
        Allows access to properties like self.model_name, self.client, etc.
        """
        return getattr(self._impl, name)

    @override
    def __repr__(self) -> str:
        return f"ModelSync(wrapping={self._impl!r})"

    @classmethod
    def create(
        cls,
        model: str | ModelSync,
        prompt: Prompt | str,
        # Configurations
        response_model: type[BaseModel] | None = None,
        project_name: str = settings.default_project_name,
        persist: bool | str = False,
        cached: bool | str = False,
        verbosity: Verbosity = settings.default_verbosity,
        console: Console | None = None,
        system: str | None = None,
        **kwargs,  # Additional kwargs for GenerationParams
    ) -> ConduitSync:
        """
        Factory method to create a ConduitSync instance with sensible defaults.
        """
        # Coerce Model and Prompt
        ## Model
        if isinstance(model, ModelSync):
            model_name = model.model_name
        elif isinstance(model, str):
            model_name = model
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        ## Prompt
        if isinstance(prompt, str):
            prompt_instance = Prompt(prompt)
        elif isinstance(prompt, Prompt):
            prompt_instance = prompt

        # Construct params and options from kwargs
        params_from_kwargs, options_from_kwargs = self._from_kwargs(kwargs)
        params = GenerationParams(**params_from_kwargs)
        options = ConduitOptions(**options_from_kwargs)

        # Repository
        repository_instance = None
        if persist:
            repository_instance: ConversationRepository = settings.default_repository(
                name=project_name
            )
        # Cache
        cache_instance = None
        if cached:
            cache_instance: settings.default_cache(name=project_name)
        # Params
        params_instance = None
        if params:
            if model_name not in params:
                params[model_name] = {}
            params[model_name].update(kwargs)
            params_instance = GenerationParams.from_dict(params[model_name])
        else:
            params_instance = GenerationParams(**params)
        conduit_options = ConduitOptions(
            verbosity=verbosity,
            cache=cache_instance,
            console=console,
        )
        return cls(
            prompt=prompt_instance,
            params=params_instance,
            options=conduit_options,
            repository=repository_instance,
            parser=parser,
        )

    # def run(..., **kwargs):
    #     param_overrides, option_overrides = self._from_kwargs(kwargs)
    #
    #     # Base params/options: instance-level defaults or global defaults
    #     base_params = self.params or GenerationParams()
    #     base_options = self.options or settings.default_conduit_options()
    #
    #     # Apply overrides on top (kwargs win)
    #     local_params = base_params.model_copy(update=param_overrides)
    #     local_options = replace(base_options, **option_overrides)
    #
    #     return self.engine.run(
    #         messages=self._make_conversation(),
    #         params=local_params,
    #         options=local_options,
    #     )
    #

    def _from_kwargs(self, kwargs: dict) -> tuple[dict, dict]:
        """
        Separate kwargs into those for GenerationParams and those for ConduitOptions.
        """
        # GenerationParams fields -- it's a BaseModel, so .schema()["properties"] works
        gen_param_fields = GenerationParams.model_fields.keys()
        params_dict = {k: v for k, v in kwargs.items() if k in gen_param_fields}
        # ConduitOptions is a @dataclass -- use its fields

        option_fields = {f.name for f in fields(ConduitOptions)}
        options_dict = {k: v for k, v in kwargs.items() if k in option_fields}
        # Check for unknown kwargs
        known = gen_param_fields | option_fields
        unknown = set(kwargs.keys()) - known
        if unknown:
            raise TypeError(f"Unknown kwargs: {sorted(unknown)}")
        return params_dict, options_dict


if __name__ == "__main__":
    from pydantic import BaseModel

    Conduit = ConduitSync

    class Frog(BaseModel):
        color: str
        legs: int
        most_embarrasing_moment: str
        year_he_received_his_helicopter_pilot_license: int
        reason_he_lost_his_helicopter_pilot_license: str
        favorite_helicopter_model: str
        no_of_ex_wives: int
        no_of_life_regrets: int
        number_of_consecutive_life_sentences: int
        favorite_netflix_show: str

    model = ModelSync("gpt4")
    prompt = Prompt("Tell me a joke about {topic}. Then design a Frog.")
    system = "You are a helpful assistant."
    conduit = Conduit(
        model=model,
        prompt=prompt,
        system=system,
        # Conduit params
        verbosity=Verbosity.COMPLETE,
        cache="test",
        persist=True,
        # Generation Params (kwargs) -- note, special handling for "model"
        response_model=Frog,
    )

    input_variables = {"topic": "helicopters"}
    response = conduit.run(input_variables=input_variables)
