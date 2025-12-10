from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message


def extract_query_preview(
    input_data: GenerationRequest | list[Message] | str, max_length: int = 100
) -> str:
    """
    Extracts a representative string from various input types for display purposes.

    Handles:
    - GenerationRequest objects (finds last user message)
    - List of Messages (finds last user message)
    - Multimodal content (Text + [Image])
    - Simple strings
    """
    content = ""

    # Case 1: It's a GenerationRequest object
    if hasattr(input_data, "messages"):
        content = _extract_from_messages(input_data.messages)

    # Case 2: It's a list of Messages
    elif isinstance(input_data, list) and input_data and hasattr(input_data[0], "role"):
        content = _extract_from_messages(input_data)

    # Case 3: It's a string or other primitive
    else:
        content = str(input_data)

    # Clean up whitespace for preview
    content = content.strip().replace("\n", " ").replace("\r", " ")

    # Truncate if necessary
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content


def _extract_from_messages(messages: list[Message]) -> str:
    """Helper to find the last user message text."""
    if not messages:
        return "No messages"

    # Walk backwards to find the last user message
    for message in reversed(messages):
        if message.role == "user":
            return _format_message_content(message)

    # Fallback to the last message if no user message found
    return _format_message_content(messages[-1])


def _format_message_content(message: Message) -> str:
    """Handle text vs multimodal content lists."""
    content = message.content

    # 1. Simple String
    if isinstance(content, str):
        return content

    # 2. Multimodal List (Text + Image/Audio)
    if isinstance(content, list):
        text_parts = []
        attachments = []

        for block in content:
            # Check Pydantic models or dicts
            block_type = getattr(block, "type", "unknown")

            if block_type == "text":
                text = getattr(block, "text", "")
                text_parts.append(text)
            elif block_type == "image_url":
                attachments.append("[Image]")
            elif block_type == "input_audio":
                attachments.append("[Audio]")

        combined = " ".join(text_parts)
        if attachments:
            combined += " " + " ".join(attachments)
        return combined

    # 3. Fallback
    return str(content)
