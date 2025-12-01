"""
Execute a Tool Call.
"""

from conduit.domain.conversation.conversation import Conversation


def execute(conversation: Conversation) -> Conversation:
    raise NotImplementedError("Tool execution is not yet implemented.")
