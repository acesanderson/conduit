"""
Display to user.
"""

from conduit.domain.conversation.conversation import Conversation


def terminate(conversation: Conversation) -> Conversation:
    raise NotImplementedError("Termination handling is not yet implemented.")
