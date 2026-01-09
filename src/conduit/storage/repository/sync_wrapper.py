"""
Synchronous wrapper for AsyncSessionRepository to maintain CLI compatibility.
"""

from __future__ import annotations
import asyncio
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.storage.repository.protocol import AsyncSessionRepository
    from conduit.domain.conversation.session import Session
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.message.message import Message


class SyncSessionRepositoryWrapper:
    """
    Synchronous wrapper around AsyncSessionRepository.
    Provides the synchronous interface needed by the CLI while delegating to async backend.
    """
    
    def __init__(self, async_repository: AsyncSessionRepository):
        self._async_repository = async_repository
    
    def initialize(self) -> None:
        """Ensure the underlying storage schema exists."""
        asyncio.run(self._async_repository.initialize())
    
    def get_session(self, session_id: str) -> Session | None:
        """Rehydrates a full Session object."""
        return asyncio.run(self._async_repository.get_session(session_id))
    
    def get_conversation(self, leaf_message_id: str) -> Conversation | None:
        """Rehydrates a linear Conversation view."""
        return asyncio.run(self._async_repository.get_conversation(leaf_message_id))
    
    def get_message(self, message_id: str) -> Message | None:
        """Fetch a single specific message by ID."""
        return asyncio.run(self._async_repository.get_message(message_id))
    
    def save_session(self, session: Session, name: str | None = None) -> None:
        """Upserts the Session metadata and all messages."""
        asyncio.run(self._async_repository.save_session(session, name))
    
    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Returns lightweight metadata for recent sessions."""
        return asyncio.run(self._async_repository.list_sessions(limit))
    
    def delete_session(self, session_id: str) -> None:
        """Hard deletes a session and all its associated messages."""
        asyncio.run(self._async_repository.delete_session(session_id))
    
    def wipe(self) -> None:
        """Hard deletes ALL sessions for the current project."""
        asyncio.run(self._async_repository.wipe())
    
    @property
    def last(self) -> Conversation | None:
        """
        Get the last conversation from the most recent session.
        This property maintains compatibility with the existing CLI interface.
        """
        sessions = self.list_sessions(limit=1)
        if not sessions:
            return None
        
        # Get the most recent session
        session_id = sessions[0]['session_id']
        session = self.get_session(session_id)
        
        if not session:
            return None
        
        # Return the conversation view of the session
        return session.conversation
    
    def load_by_conversation_id(self, conversation_id: str) -> Conversation | None:
        """
        Load conversation by ID - maintains compatibility with old interface.
        Note: In the new architecture, conversation_id is the same as session_id.
        """
        session = self.get_session(conversation_id)
        return session.conversation if session else None
    
    def remove_by_conversation_id(self, conversation_id: str) -> None:
        """
        Remove conversation by ID - maintains compatibility with old interface.
        Note: In the new architecture, conversation_id is the same as session_id.
        """
        self.delete_session(conversation_id)
    
    def save(self, conversation: Conversation, name: str = "Untitled") -> None:
        """
        Save a conversation - maintains compatibility with old interface.
        Note: This converts the Conversation back to a Session for storage.
        """
        from conduit.domain.conversation.session import Session
        
        # Create a new session from the conversation
        session = Session()
        
        # Add all messages from the conversation to the session
        for message in conversation.messages:
            session.add_message(message)
        
        self.save_session(session, name)