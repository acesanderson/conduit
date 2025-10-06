from collections.abc import Iterator
from pydantic import BaseModel, Field, ValidationError
from conduit.message.message import Message
from conduit.message.textmessage import TextMessage
from conduit.message.imagemessage import ImageMessage
from conduit.message.audiomessage import AudioMessage
from conduit.logs.logging_config import get_logger
from typing import override

MessageUnion = TextMessage | ImageMessage | AudioMessage

logger = get_logger(__name__)


class Messages(BaseModel):
    """
    A Pydantic BaseModel that contains a list of Message objects.
    Behaves like a list through dunder methods while being fully Pydantic-compatible.
    Supports Message, ImageMessage, and AudioMessage objects (all inherit from Message).
    class MessageStore(Messages):
    """

    messages: list[MessageUnion | None] = Field(
        default_factory=list,
        description="List of Message objects (including ImageMessage and AudioMessage)",
    )

    def __init__(self, messages: list[Message] = [], **kwargs):
        """
        Initialize with optional list of messages.

        Args:
            messages: list of Message objects (including ImageMessage and AudioMessage) to initialize with
        """
        if messages is not None:
            super().__init__(messages=messages, **kwargs)
        else:
            super().__init__(**kwargs)
        # Validate the turn order of messages
        self._validate_turn_order()

    # Validation methods
    def _validate_turn_order(self):
        """
        Validate that the message history meets the following conditions:
        (1) only one system message
        (2) messages alternate between user and assistant roles

        This is called on init and whenever messages are added.
        """

        def create_dialog_signature(messages: list[Message]) -> str:
            """
            Create a signature string for the message order.
            """
            return "".join(msg.role[0].lower() for msg in messages)

        dialog_signature = create_dialog_signature(self.messages)
        # Check for exactly one system message
        if dialog_signature.count("s") > 1:
            raise ValueError(
                "Only one 'system' message is allowed in the conversation history."
            )
        # Check for alternating user and assistant messages
        if "uu" in dialog_signature or "aa" in dialog_signature:
            raise ValueError(
                "Messages must alternate between 'user' and 'assistant' roles."
            )
        pass

    # List-like interface methods
    def append(self, message: Message) -> None:
        """Add a message to the end of the list."""
        self.messages.append(message)
        self._validate_turn_order()

    def extend(self, messages: list[Message]) -> None:
        """Extend the list with multiple messages."""
        self.messages.extend(messages)

    def insert(self, index: int, message: Message) -> None:
        """Insert a message at the specified index."""
        self.messages.insert(index, message)

    def remove(self, message: Message) -> None:
        """Remove the first occurrence of a message."""
        self.messages.remove(message)

    def pop(self, index: int = -1) -> Message:
        """Remove and return message at index (default last)."""
        return self.messages.pop(index)

    def clear(self) -> None:
        """Remove all messages."""
        self.messages.clear()

    def index(self, message: Message, start: int = 0, stop: int | None = None) -> int:
        """Return the index of the first occurrence of message."""
        if stop is None:
            return self.messages.index(message, start)
        return self.messages.index(message, start, stop)

    def count(self, message: Message) -> int:
        """Return the number of occurrences of message."""
        return self.messages.count(message)

    def reverse(self) -> None:
        """Reverse the messages in place."""
        self.messages.reverse()

    # Dunder methods for list-like behavior
    @override
    def __repr__(self) -> str:
        """String representation for debugging."""
        return str(self.messages)

    @override
    def __str__(self) -> str:
        """
        String representation showing message count and types.
        """
        return self.__repr__()

    @override
    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages."""
        return iter(self.messages)

    @override
    def __eq__(self, other) -> bool:
        """Check equality with another Messages object or list."""
        if isinstance(other, Messages):
            return self.messages == other.messages
        elif isinstance(other, list):
            return self.messages == other
        return False

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __getitem__(self, key) -> "Message | Messages":
        """Get message(s) by index or slice."""
        result = self.messages[key]
        if isinstance(key, slice):
            return Messages(result)
        return result

    def __setitem__(self, key, value):
        """Set message(s) by index or slice."""
        self.messages[key] = value

    def __delitem__(self, key):
        """Delete message(s) by index or slice."""
        del self.messages[key]

    def __reversed__(self) -> Iterator[Message]:
        """Iterate over messages in reverse order."""
        return reversed(self.messages)

    def __contains__(self, message: Message) -> bool:
        """Check if message is in the list."""
        return message in self.messages

    def __bool__(self) -> bool:
        """Return True if there are any messages."""
        return bool(self.messages)

    def __add__(self, other) -> "Messages":
        """Concatenate with another Messages object or list."""
        if isinstance(other, Messages):
            return Messages(self.messages + other.messages)
        elif isinstance(other, list):
            return Messages(self.messages + other)
        return NotImplemented

    def __iadd__(self, other) -> "Messages":
        """In-place concatenation with another Messages object or list."""
        if isinstance(other, Messages):
            self.messages.extend(other.messages)
        elif isinstance(other, list):
            self.messages.extend(other)
        else:
            return NotImplemented
        return self

    def __mul__(self, other: int) -> "Messages":
        """Repeat messages n times."""
        if isinstance(other, int):
            return Messages(self.messages * other)
        return NotImplemented

    def __imul__(self, other: int) -> "Messages":
        """In-place repeat messages n times."""
        if isinstance(other, int):
            self.messages *= other
        else:
            return NotImplemented
        return self

    # Conduit-specific convenience methods
    def add_new(self, role: str, content: str) -> None:
        """
        Create and add a new message to the list.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        if role == "system" and self.system_message is not None:
            raise ValueError(
                "Only one 'system' message is allowed in the conversation."
            )
        else:
            self.append(TextMessage(role=role, content=content))

    def last(self) -> Message | None:
        """
        Get the last message in the list.

        Returns:
            Last Message object or None if empty
        """
        return self.messages[-1] if self.messages else None

    def get_by_role(self, role: str) -> list[Message]:
        """
        Get all messages with a specific role.

        Args:
            role: Role to filter by (user, assistant, system)

        Returns:
            list of messages with the specified role
        """
        return [msg for msg in self.messages if msg.role == role]

    def user_messages(self) -> list[Message]:
        """Get all user messages."""
        return self.get_by_role("user")

    def assistant_messages(self) -> list[Message]:
        """Get all assistant messages."""
        return self.get_by_role("assistant")

    def system_messages(self) -> list[Message]:
        """Get all system messages."""
        return self.get_by_role("system")

    @property
    def system_message(self) -> Message | None:
        """Get the system message, if any."""
        systems = self.get_by_role("system")
        return systems[0] if systems else None

    # Serialization methods
    def to_cache_dict(self) -> list:
        """
        This differs from our usual serialization to_cache_dict method in two ways:
        - It returns a list of dictionaries, one for each message (not a dict)
        - It requires a provider argument, which is used to determine how to serialize the messages.

        This is because audio and image messages may have different serialization requirements per provider.
        """
        return [msg.to_cache_dict() for msg in self.messages]

    @classmethod
    def from_cache_dict(cls, cache_dict: list) -> "Messages":
        """
        Deserialize from a dictionary. Note: in most cases we have serialized to a list of dictionaries, though this can also handle a single dictionary (i.e. the full Messages object).

        Args:
            cache_dict: Dictionary containing cached messages

        Returns:
            Messages object
        """
        if isinstance(cache_dict, list):
            # Convert each dict to proper Message object
            message_dicts = [Message.from_cache_dict(msg) for msg in cache_dict]
            return cls(messages=message_dicts)
        else:
            # If this assertion breaks, we are deserializing a full Messages object, not a list of messages.
            raise ValueError("cache_dict must be a list of message dictionaries")

    # API compatibility methods
    def to_openai(self) -> list[dict]:
        return [msg.to_openai() for msg in self.messages]

    def to_anthropic(self) -> list[dict]:
        return [msg.to_anthropic() for msg in self.messages]

    def to_google(self) -> list[dict]:
        return [msg.to_google() for msg in self.messages]

    def to_ollama(self) -> list[dict]:
        return [msg.to_ollama() for msg in self.messages]

    def to_perplexity(self) -> list[dict]:
        return [msg.to_perplexity() for msg in self.messages]
