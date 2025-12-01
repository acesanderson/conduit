from conduit.core.model.clients.openai.adapter import convert_message_to_openai
from conduit.domain.message.message import Message
from typing import Any


def convert_message_to_ollama(message: Message) -> dict[str, Any]:
    return convert_message_to_openai(message)
