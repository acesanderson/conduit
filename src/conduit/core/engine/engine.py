"""
Our Engine takes a Conversation and runs it through an LLM or other processing.
Think of it as a FSM that processes the conversation.
LLMs produce the next token; Conduit produces the next Message.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.domain.conversation.conversation import (
    Conversation,
    ConversationError,
    ConversationState,
)
from conduit.domain.request.request import Request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.request.generation_params import GenerationParams


class Engine:
    def __init__(self, params: GenerationParams = settings.default_params) -> None:
        self.params: GenerationParams = params

    # Main entry point
    def run(self, conversation: Conversation) -> Conversation:
        if not conversation.last:
            raise ConversationError("Conversation is empty.")

        match conversation.state:
            case ConversationState.GENERATE:
                return self._generate(conversation)
            case ConversationState.EXECUTE:
                return self._execute(conversation)
            case ConversationState.TERMINATE:
                return self._terminate(conversation)
            case ConversationState.INCOMPLETE:
                raise ConversationError("Conversation is incomplete.")

    # Handlers
    def _generate(self, conversation: Conversation) -> Conversation:
        from conduit.core.engine.generate import generate

        return generate(conversation, params=conversation.metadata or self.params)

    def _execute(self, conversation: Conversation) -> Conversation:
        from conduit.core.engine.execute import execute

        return execute(conversation)

    def _terminate(self, conversation: Conversation) -> Conversation:
        from conduit.core.engine.terminate import terminate

        return terminate(conversation)

    # Utility Methods
    def _assemble_request(self, conversation: Conversation) -> Request:
        """
        Assemble a Request object from the Conversation and GenerationParams.
        """
        params = conversation.metadata or self.params  # Cascade params
        params_dict = params.model_dump()
        request = Request(messages=conversation.messages, **params_dict)
        return request
