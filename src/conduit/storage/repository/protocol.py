from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, TYPE_CHECKING

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
        """
        Ensure the underlying storage schema exists.
        """
        ...

    async def get_session(self, session_id: str) -> Session | None:
        """
        Rehydrates a full Session object, including the entire message graph
        (message_dict) associated with it.
        """
        ...

    async def get_conversation(self, leaf_message_id: str) -> Conversation | None:
        """
        Rehydrates a linear Conversation view by walking backwards from a specific leaf.
        """
        ...

    async def get_message(self, message_id: str) -> Message | None:
        """
        Fetch a single specific message by ID.
        """
        ...

    async def save_session(self, session: Session, name: str | None = None) -> None:
        """
        Upserts the Session metadata and *all* messages currently in the session.
        Uses topological sorting to ensure referential integrity.
        """
        ...

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Returns lightweight metadata for recent sessions in this project.
        """
        ...

    async def delete_session(self, session_id: str) -> None:
        """
        Hard deletes a session and all its associated messages.
        """
        ...

    async def wipe(self) -> None:
        """
        Hard deletes ALL sessions for the current project.
        """
        ...
