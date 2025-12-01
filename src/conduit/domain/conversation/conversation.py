"""
A Conversation wraps a list[Message] with extra metadata, validation, and helper methods.
"""

from __future__ import annotations


class Conversation:
    def __init__(self, messages=None, title=None):
        self.messages = messages if messages is not None else []
        self.title = title

    def add_message(self, message):
        self.messages.append(message)

    def add_new(self, role, content):
        message = TextMessage(role=role, content=content)
        self.add_message(message)

    @property
    def last(self) -> Message:
        if self.messages:
            return self.messages[-1]
        return None

    def validate(self):
        if not isinstance(self.messages, list):
            raise ValueError("Messages must be a list")
        for message in self.messages:
            if not isinstance(message, Message):
                raise ValueError("All items in messages must be of type Message")

    def __repr__(self):
        return f"Conversation(title={self.title}, messages={self.messages})"
