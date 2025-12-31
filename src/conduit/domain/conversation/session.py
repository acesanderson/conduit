from pydantic import BaseModel, Field
from conduit.domain.message.message import Message
import time


class Session(BaseModel):
    """
    Session is both a runtime and persistent object.
    It holds all messages generated in a session (the "Graph").
    """

    session_id: str = Field(..., description="Unique identifier for the session")
    message_dict: dict[str, Message] = Field(
        ..., description="Mapping of message_id to Message objects"
    )
    leaf: str = Field(
        ..., description="The latest message id in the session -- for persistence"
    )
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))

    def register(self, message: Message) -> None:
        """
        Idempotent registration of a message into the session graph.
        Updates the session's leaf pointer to this new message.
        """
        # 1. Add to graph (Idempotent)
        self.message_dict[message.message_id] = message

        # 2. Update Cursor
        # In a strict tree append, the new message is always the new leaf.
        self.leaf = message.message_id
