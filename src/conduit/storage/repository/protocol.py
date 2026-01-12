from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, TYPE_CHECKING
from collections.abc import Awaitable

if TYPE_CHECKING:
    from conduit.domain.conversation.session import Session
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.message.message import Message


@runtime_checkable
class AsyncSessionRepository(Protocol):
    """
    Async interface for persisting Conduit sessions (DAGs) and conversations (Linear Views).
    Strictly scoped to a specific 'project_name'.
    """

    async def initialize(self) -> None:
        """Ensure the underlying storage schema exists."""
        ...

    @property
    def last(self) -> Awaitable[Conversation | None]:
        """
        Retrieves the most recently updated conversation for this project.
        """
        ...

    async def get_session(self, session_id: str) -> Session | None:
        """Rehydrates a full Session object."""
        ...

    async def get_conversation(self, leaf_message_id: str) -> Conversation | None:
        """Rehydrates a linear Conversation view."""
        ...

    async def get_message(self, message_id: str) -> Message | None:
        """Fetch a single specific message by ID."""
        ...

    async def save_session(self, session: Session, name: str | None = None) -> None:
        """Upserts the Session metadata and all messages."""
        ...

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Returns lightweight metadata for recent sessions."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Hard deletes a session."""
        ...

    async def wipe(self) -> None:
        """Hard deletes ALL sessions for the current project."""
        ...
