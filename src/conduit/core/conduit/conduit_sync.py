"""
NOTE: Refactor incoming -- ConduitSync should LITERALLY just be ConduitAsync wrapped with asyncio.run

from conduit.utils.concurrency.warn import _warn_if_loop_exists

class ConduitSync:
    def __init__(self):
        self._impl = ConduitAsync()

    def pipe(self, payload: list[dict]) -> bool:
        _warn_if_loop_exists()
        return asyncio.run(self._impl.pipe(payload))
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.message.message import UserMessage
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.core.model.model_sync import ModelSync
    from conduit.core.prompt.prompt import Prompt
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.core.engine.engine import Engine
    from conduit.storage.repository.protocol import (
        ConversationRepository,
    )


class ConduitSync:
    """
    A Configured Pipe.
    State regarding 'How to process' is held in self (Prompt, Params).
    State regarding 'What to process' is passed to run().
    """

    def __init__(
        self,
        # Required
        prompt: Prompt,
        params: GenerationParams,
        options: ConduitOptions,
    ):
        # Initial attributes
        self.prompt: Prompt = prompt
        self.params: GenerationParams = params
        self.options: ConduitOptions = options

    @classmethod
    def create(
        cls,
        model: str | ModelSync,
        prompt: Prompt | str,
        params: dict[str, str | dict[str, str]] | None = None,
        # Configurations
        response_model: type | None = None,
        project_name: str = settings.default_project_name,
        persist: bool = False,
        cached: bool = False,
        verbosity: Verbosity = settings.default_verbosity,
        console: Console | None = None,
        **kwargs,  # Additional kwargs for GenerationParams
    ) -> ConduitSync:
        """
        Factory method to create a ConduitSync instance with sensible defaults.
        """
        # Model
        if isinstance(model, ModelSync):
            model_name = model.model_name
        elif isinstance(model, str):
            model_name = model
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        # Prompt
        if isinstance(prompt, str):
            prompt_instance = Prompt(prompt)
        elif isinstance(prompt, Prompt):
            prompt_instance = prompt
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

    # Conversation management
    def _execute_with_engine(
        self,
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        """
        Given a conversation, execute it using an Engine.
        If needed, can take decorators for telemetry, logging, progress, caching, etc.
        """
        new_conversation = Engine.run(conversation, params, options)
        return new_conversation

    async def _execute_with_engine_async(
        self,
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        """
        Given a conversation, execute it using an Engine.
        If needed, can take decorators for telemetry, logging, progress, caching, etc.
        """
        new_conversation = await Engine.run_async(conversation, params, options)
        return new_conversation

    def _update_topic(self, conversation: Conversation) -> Conversation:
        """
        Create a title for the current conversation based on the prompt.
        Will leverage an external service.
        """
        raise NotImplementedError("_create_title not yet implemented.")

    async def _update_topic_async(self, conversation: Conversation) -> Conversation:
        """
        Create a title for the current conversation based on the prompt.
        Will leverage an external service.
        """
        raise NotImplementedError("_create_title not yet implemented.")

    def run(
        self,
        input_variables: dict[str, str] | None = None,
        stream: bool = False,
        # Progress tracking
        index: int = 0,
        total: int = 0,
        # Possible overrides
        cached: bool = True,
        persist: bool = True,
        include_history: bool = True,
        verbosity: Verbosity | None = None,
    ) -> Conversation:
        # Validate and render prompt
        if input_variables and self.prompt:
            try:
                rendered = self.prompt.render(input_variables=input_variables)
            except Exception as e:
                logger.error(
                    "Error rendering prompt with input variables %s: %s",
                    input_variables,
                    e,
                )
                raise
            self.prompt.validate_input_variables(input_variables)
            logger.info("Rendering prompt with input variables: %s", input_variables)
            rendered_prompt = self.prompt.render(input_variables=input_variables)
        elif self.prompt:
            logger.info("Using prompt without input variables.")
            rendered_prompt = self.prompt.prompt_string
        # Load conversation or create new one
        if persist and self.options.repository and self.options.repository.last:
            logger.info("Loading or creating conversation from repository.")
            conversation = self.options.repository.last
        else:
            logger.info("Creating new conversation.")
            conversation = Conversation()
        # Append rendered prompt to conversation as UserMessage
        logger.info("Appending rendered prompt to conversation.")
        conversation.messages.append(UserMessage(content=rendered_prompt))
        # Execute conversation with engine
        logger.info("Executing conversation with engine.")
        updated_conversation = self._execute_with_engine(
            conversation=conversation,
            params=self.params,
            options=self.options,
        )
        # Update topic if needed
        if not updated_conversation.topic or updated_conversation.topic == "Untitled":
            logger.info("Updating conversation topic.")
            updated_conversation.topic = self._update_topic()
        # Save conversation if needed
        if persist and self.options.repository:
            logger.info("Saving conversation to repository.")
            self.options.repository.save(updated_conversation)
        return updated_conversation
