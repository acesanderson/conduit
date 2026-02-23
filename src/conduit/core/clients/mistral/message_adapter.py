from __future__ import annotations
from conduit.core.clients.openai.message_adapter import convert_message_to_openai
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conduit.domain.message.message import Message


def convert_message_to_mistral(message: Message) -> dict[str, Any]:
    return convert_message_to_openai(message)
