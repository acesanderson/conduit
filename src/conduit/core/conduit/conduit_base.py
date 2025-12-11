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
    from conduit.core.prompt.prompt import Prompt
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.core.engine.engine import Engine
    from conduit.domain.message.message import Message


class ConduitBase:
    """
    A dumb pipe.
    """

    def __init__(
        self,
        # Required
        prompt: Prompt,
    ):
        # Initial attributes
        self.prompt: Prompt = prompt

    # Our "pipe" to Engine
    async def pipe(
        self,
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        """
        Given a conversation, execute it using an Engine.
        If needed, can take decorators for telemetry, logging, progress, caching, etc.
        """
        new_conversation = await Engine.run(conversation, params, options)
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

    async def run(
        self,
        input_variables: dict[str, str] | None = None,
        messages: list[Message] | None = None,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
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
