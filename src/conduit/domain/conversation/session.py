from __future__ import annotations
from pydantic import BaseModel, Field
from conduit.domain.message.message import Message
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation


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

    def to_conversation(
        self, message_id: str, ensure_leaf: bool = True
    ) -> Conversation:
        """
        Reconstruct a conversation for a given leaf (by message_id).
        """
        from conduit.domain.conversation.conversation import Conversation

        if message_id not in self.message_dict:
            raise KeyError(f"Message ID {message_id} not found in session.")

        # If ensure_leaf, verify that the requested message_id is a TERMINATING message (no children).
        if ensure_leaf:
            is_leaf = all(
                msg.predecessor_id != message_id for msg in self.message_dict.values()
            )
            if not is_leaf:
                raise ValueError(f"Message ID {message_id} is not a leaf message.")

        # Reconstruct the message chain
        messages = []
        current_id = message_id
        while current_id:
            msg = self.message_dict[current_id]
            messages.append(msg)
            current_id = msg.predecessor_id
            if current_id == message_id:
                raise ValueError(
                    "Conversation is an infinite loop due to circular reference."
                )
        messages.reverse()  # Reverse to get chronological order
        return Conversation(messages=messages)

    def branch(self, from_message_id: str) -> Conversation:
        """
        Create a new conversation branch from an existing message in the session.
        """
        if from_message_id not in self.message_dict:
            raise KeyError(f"Message ID {from_message_id} not found in session.")

        message = self.message_dict[from_message_id]
        if message.predecessor_id is None:
            raise ValueError("Cannot branch from a root message.")
        if message.predecessor_id not in self.message_dict:
            raise ValueError(
                "Cannot branch from a message whose predecessor is not in the session."
            )

        return self.to_conversation(from_message_id, ensure_leaf=False)

    @property
    def conversation(self) -> Conversation:
        """
        Get the conversation for the current leaf message.
        """
        return self.to_conversation(self.leaf, ensure_leaf=True)

    @property
    def leaves(self) -> list[Message]:
        """
        Get all leaf messages in the session.
        """
        leaf_messages = []
        for msg in self.message_dict.values():
            is_leaf = all(
                other_msg.predecessor_id != msg.message_id
                for other_msg in self.message_dict.values()
            )
            if is_leaf:
                leaf_messages.append(msg)
        return leaf_messages

    @property
    def conversations(self) -> list[Conversation]:
        """
        Get all conversations for each leaf in the session.
        """
        convos = []
        for leaf_msg in self.leaves:
            convo = self.to_conversation(leaf_msg.message_id, ensure_leaf=True)
            convos.append(convo)
        return convos
