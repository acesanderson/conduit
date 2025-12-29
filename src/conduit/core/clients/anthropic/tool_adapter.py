from __future__ import annotations
from typing import Any
from conduit.capabilities.tools.tool import Tool


def convert_tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    """
    Convert a canonical Tool to Anthropic's tool format.

    Anthropic's format:
    {
        "name": "function_name",
        "description": "function description",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

    Note: Unlike OpenAI, Anthropic doesn't wrap tools in a "function" key,
    and uses "input_schema" directly (not "parameters").
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema.model_dump(
            by_alias=True, exclude_none=True
        ),
    }
