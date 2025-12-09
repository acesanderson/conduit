"""
Our Engine takes a Conversation and runs it through an LLM or other processing.
Think of it as a FSM that processes the conversation.
LLMs produce the next token; Conduit produces the next Message.
"""

from __future__ import annotations
from conduit.domain.conversation.conversation import (
    Conversation,
    ConversationError,
    ConversationState,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions


class Engine:
    # Main entry point

    @staticmethod
    def run(
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
        max_steps: int = 10,  # Safety limit for auto-looping
    ) -> Conversation:
        """
        This is pure CPU, so no async needed (though the handlers may need async).
        """
        step_count = 0

        while step_count < max_steps:
            if not conversation.last:
                raise ConversationError("Conversation is empty.")

            match conversation.state:
                # 1. LLM Generation
                case ConversationState.GENERATE:
                    conversation = Engine._generate(conversation, params, options)

                # 2. Tool Execution (The Loop back)
                case ConversationState.EXECUTE:
                    conversation = Engine._execute(conversation, params, options)

                # 3. Stop Conditions
                case ConversationState.TERMINATE:
                    return Engine._terminate(conversation, params, options)

                case ConversationState.INCOMPLETE:
                    raise ConversationError("Conversation is incomplete.")

            step_count += 1

        # If we exit the loop, we hit the limit
        logger.warning(f"Engine hit max_steps ({max_steps}). returning current state.")
        return conversation

    # Handlers
    @staticmethod
    def _generate(
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        from conduit.core.engine.generate import generate

        return generate(conversation, params, options)

    @staticmethod
    def _execute(
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        from conduit.core.engine.execute import execute

        return execute(conversation, params, options)

    @staticmethod
    def _terminate(
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> Conversation:
        from conduit.core.engine.terminate import terminate

        return terminate(conversation, params, options)
