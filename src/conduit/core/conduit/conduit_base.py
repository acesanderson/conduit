from __future__ import annotations
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.message.message import UserMessage
from conduit.core.prompt.prompt import Prompt
from typing import Any, override
import logging

logger = logging.getLogger(__name__)


class ConduitBase:
    """
    Stem class for Conduit implementations; not to be used directly.
    Holds shared logic for conversation orchestration via Engine.
    """

    def __init__(self, prompt: Prompt):
        """
        Initialize the Conduit base with only its identity.

        Args:
            prompt: The template/prompt configuration (the unique identity of this Conduit)
        """
        self.prompt: Prompt = prompt

    # Core delegation (the "dumb pipe")
    async def pipe(
        self,
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        """
        Given a conversation, execute it using Engine.
        This is the core delegation point - analogous to Model.pipe().
        """
        from conduit.core.engine.engine import Engine

        return await Engine.run(conversation, params, options)

    # Pure CPU helper methods
    def _render_prompt(self, input_variables: dict[str, Any] | None) -> str:
        """
        PURE CPU: Render the prompt template with input variables.
        """
        if input_variables:
            self.prompt.validate_input_variables(input_variables)
            return self.prompt.render(input_variables=input_variables)
        return self.prompt.prompt_string

    def _prepare_conversation(
        self, rendered_prompt: str, params: GenerationParams, options: ConduitOptions
    ) -> Conversation:
        """
        PURE CPU: Build initial conversation object.
        Load from repository if enabled and last conversation exists.
        """
        # Load from repository if persistence is enabled
        if options.repository and options.repository.last:
            logger.info("Loading last conversation from repository.")
            conversation = options.repository.last
        else:
            logger.info("Creating new conversation.")
            conversation = Conversation()

        # Append rendered prompt as UserMessage
        conversation.messages.append(UserMessage(content=rendered_prompt))
        # Attach system message if exists in options
        if params.system:
            conversation.ensure_system_message(params.system)
        return conversation

    # Abstract methods (must be implemented by subclasses)
    async def run(
        self,
        input_variables: dict[str, Any] | None,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        """
        Main entry point for executing the Conduit.
        Must be implemented by subclasses (sync or async).

        Args:
            input_variables: Template variables for the prompt
            params: Generation parameters (model, temperature, etc.)
            options: Conduit options (cache, repository, console, etc.)

        Returns:
            Conversation: The completed conversation after execution
        """
        raise NotImplementedError(
            "run must be implemented in subclasses (sync or async)."
        )

    # Dunders
    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prompt={self.prompt!r})"
