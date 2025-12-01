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


def convert_message_to_openai(message: Message) -> dict[str, Any]:
    """
    Pure function to adapt an internal Message DTO into an OpenAI-compatible dictionary.
    Used by OpenAI, Ollama, Google, Perplexity, and Groq clients.
    """
    match message:
        # 1. System Message
        case SystemMessage(content=content):
            return {"role": "system", "content": content}

        # 2. User Message (Text or Multimodal)
        case UserMessage(content=content, name=name):
            payload = {"role": "user"}
            if name:
                payload["name"] = name

            if isinstance(content, str):
                payload["content"] = content
                return payload

            # Handle list of content blocks (Multimodal)
            openai_content = []
            for block in content:
                match block:
                    case TextContent(text=text):
                        openai_content.append({"type": "text", "text": text})
                    case ImageContent(url=url, detail=detail):
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url, "detail": detail},
                            }
                        )
                    case AudioContent(data=data, format=fmt):
                        # GPT-4o-audio format for input audio
                        openai_content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {"data": data, "format": fmt},
                            }
                        )
            payload["content"] = openai_content
            return payload

        # 3. Assistant Message (Text, Reasoning, Tools, Audio)
        case AssistantMessage(content=content, tool_calls=calls, audio=audio):
            payload = {"role": "assistant"}

            # OpenAI allows null content if tool_calls are present
            if content:
                payload["content"] = content
            elif not calls and not audio:
                # If everything is empty (rare), avoid sending null
                payload["content"] = ""

            # Handle Tool Calls
            if calls:
                payload["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function_name,
                            # OpenAI mandates arguments be a JSON string, not a dict
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in calls
                ]

            # Handle Audio Response Context (sending audio back to model)
            if audio:
                payload["audio"] = {"id": audio.id}

            return payload

        # 4. Tool Result
        case ToolMessage(content=result, tool_call_id=call_id):
            return {"role": "tool", "content": result, "tool_call_id": call_id}

        case _:
            raise ValueError(
                f"Unknown message type for OpenAI Adapter: {type(message)}"
            )
