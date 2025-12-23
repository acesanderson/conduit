from __future__ import annotations
from typing import Protocol, runtime_checkable
from uuid import UUID
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation


@runtime_checkable
class ConversationRepository(Protocol):
    """
    The Playlist Manager.
    Saves and Loads Conversations as sequences of Message IDs.
    """

    def load_by_conversation_id(
        self, conversation_id: str | UUID
    ) -> Conversation | None:
        """Rehydrates a Conversation object from the DB."""
        ...

    def load_by_name(self, name: str) -> Conversation | None:
        """Loads a Conversation by its given name, if any."""
        ...

    def list_conversations(self, limit: int = 10) -> list[dict[str, str]]:
        """Lists Conversations with metadata only."""
        ...

    def load_all(self) -> list[Conversation]:
        """Loads all Conversations in the repository."""
        ...

    def save(self, conversation: Conversation, name: str | None = None) -> None:
        """Upserts Conversation metadata and message links."""
        ...

    def remove_by_conversation_id(self, conversation_id: str | UUID) -> None:
        """Removes a Conversation and its links from the repository."""
        ...

    def wipe(self) -> None:
        """Wipes all Conversations from the repository."""
        ...

    @property
    def last(self) -> Conversation | None:
        """Returns the most recently used Conversation, if any."""
        ...
