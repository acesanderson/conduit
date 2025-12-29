from conduit.core.clients.openai.tool_adapter import convert_tool_to_openai
from conduit.capabilities.tools.tool import Tool
from typing import Any


def convert_tool_to_ollama(tool: Tool) -> dict[str, Any]:
    return convert_tool_to_openai(tool)
