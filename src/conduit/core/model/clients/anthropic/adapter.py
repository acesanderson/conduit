from __future__ import annotations
from typing import Any
from conduit.domain.message.message import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    TextContent,
    ImageContent,
    AudioContent,
)
import json
import re


def convert_message_to_anthropic(message: Message) -> dict[str, Any]:
    """
    Pure function to adapt an internal Message DTO into an Anthropic-compatible dictionary.
    
    Key differences from OpenAI:
    - Images use {"type": "image", "source": {...}} instead of image_url
    - Audio is not supported
    - System messages are converted to user messages here; extraction happens at request level
    """
    match message:
        # 1. System Message
        # Anthropic doesn't support system role in messages array
        # This will be filtered out in _convert_request
        case SystemMessage(content=content):
            return {"role": "user", "content": content}
        
        # 2. User Message (Text or Multimodal)
        case UserMessage(content=content, name=name):
            payload = {"role": "user"}
            
            if isinstance(content, str):
                payload["content"] = content
                return payload
            
            # Handle list of content blocks (Multimodal)
            anthropic_content = []
            for block in content:
                match block:
                    case TextContent(text=text):
                        anthropic_content.append({"type": "text", "text": text})
                    case ImageContent(url=url, detail=_):
                        # Parse base64 data URL for Anthropic format
                        # Expected format: data:image/png;base64,iVBORw0KG...
                        image_data = _parse_image_url(url)
                        anthropic_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_data["media_type"],
                                    "data": image_data["data"],
                                },
                            }
                        )
                    case AudioContent(data=data, format=fmt):
                        raise NotImplementedError(
                            "Anthropic API does not support audio input messages."
                        )
            payload["content"] = anthropic_content
            return payload
        
        # 3. Assistant Message (Text, Reasoning, Tools)
        case AssistantMessage(content=content, tool_calls=calls, audio=audio):
            payload = {"role": "assistant"}
            
            # Anthropic requires content to be present
            if content:
                payload["content"] = content
            elif calls:
                # Tool-only response needs empty content array
                payload["content"] = []
            else:
                payload["content"] = ""
            
            # Handle Tool Calls
            if calls:
                # Anthropic uses 'tool_use' blocks within content
                if not isinstance(payload["content"], list):
                    # Convert string content to text block
                    payload["content"] = [{"type": "text", "text": payload["content"]}]
                
                for call in calls:
                    payload["content"].append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function_name,
                            "input": call.arguments,  # Anthropic uses dict, not JSON string
                        }
                    )
            
            # Audio output is not supported
            if audio:
                raise NotImplementedError(
                    "Anthropic API does not support audio output messages."
                )
            
            return payload
        
        # 4. Tool Result
        case ToolMessage(content=result, tool_call_id=call_id):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": result,
                    }
                ],
            }
        
        case _:
            raise ValueError(
                f"Unknown message type for Anthropic Adapter: {type(message)}"
            )


def _parse_image_url(url: str) -> dict[str, str]:
    """
    Parse a data URL to extract media type and base64 data.
    
    Expected format: data:image/png;base64,iVBORw0KG...
    Returns: {"media_type": "image/png", "data": "iVBORw0KG..."}
    """
    if url.startswith("data:"):
        # Parse data URL
        match = re.match(r"data:([^;]+);base64,(.+)", url)
        if match:
            media_type, data = match.groups()
            return {"media_type": media_type, "data": data}
        raise ValueError(f"Invalid data URL format: {url}")
    else:
        # External URL - Anthropic doesn't support external image URLs directly
        raise ValueError(
            "Anthropic requires base64-encoded images, not external URLs. "
            "Please convert the image to base64 first."
        )
