from conduit.core.clients.openai.message_adapter import convert_message_to_openai
from conduit.domain.message.message import Message
from typing import Any


def convert_message_to_perplexity(message: Message) -> dict[str, Any]:
    return convert_message_to_openai(message)
