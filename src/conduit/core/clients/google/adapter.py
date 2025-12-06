from conduit.core.clients.openai.adapter import convert_message_to_openai
from conduit.domain.message.message import Message
from typing import Any


def convert_message_to_google(message: Message) -> dict[str, Any]:
    return convert_message_to_openai(message)
