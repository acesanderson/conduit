from typing import Protocol, runtime_checkable
from conduit.domain.conversation.conversation import Conversation
from uuid import UUID


@runtime_checkable
class ConversationRepository(Protocol):
    """
    The Playlist Manager.
    Saves and Loads Conversations as sequences of Message IDs.
    """

    def load(
        self, conversation_id: str | UUID, name: str | None = None
    ) -> Conversation | None:
        """Rehydrates a Conversation object from the DB."""
        ...

    def save(self, conversation: Conversation, name: str | None = None) -> None:
        """Upserts Conversation metadata and message links."""
        ...

    @property
    def last(self) -> Conversation | None:
        """Returns the most recently used Conversation, if any."""
        ...
