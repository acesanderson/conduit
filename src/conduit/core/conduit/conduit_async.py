from __future__ import annotations
from conduit.core.conduit.conduit_base import ConduitBase
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from typing import override, TYPE_CHECKING, Any
import logging

if TYPE_CHECKING:
    from conduit.core.prompt.prompt import Prompt

logger = logging.getLogger(__name__)


class ConduitAsync(ConduitBase):
    """
    Async implementation of Conduit - a stateless "dumb pipe".
    Execution context (params/options) passed explicitly to run().
    """

    @override
    async def run(
        self,
        input_variables: dict[str, Any] | None,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        """
        Execute the Conduit with explicit execution context.

        Args:
            input_variables: Template variables for the prompt
            params: Generation parameters (model, temperature, etc.)
            options: Conduit options (cache, repository, console, etc.)

        Returns:
            Conversation: The completed conversation after execution
        """
        # 1. Render prompt
        rendered = self._render_prompt(input_variables)

        # 2. Prepare conversation (may load from repository)
        conversation = await self._prepare_conversation(rendered, params, options)

        # 3. Execute via Engine
        updated_conversation = await self.pipe(conversation, params, options)

        # 4. Save if repository is configured
        if options.repository:
            logger.info("Saving conversation to repository.")
            if updated_conversation.session:
                await options.repository.save_session(
                    updated_conversation.session, name=updated_conversation.topic
                )
            else:
                logger.warning(
                    "Conversation has no session initialized; skipping persistence."
                )

        return updated_conversation
