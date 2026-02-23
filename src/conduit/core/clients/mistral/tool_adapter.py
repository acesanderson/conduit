from __future__ import annotations
from conduit.core.clients.openai.tool_adapter import convert_tool_to_openai
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.capabilities.tools.tool import Tool


def convert_tool_to_mistral(tool: Tool) -> dict:
    return convert_tool_to_openai(tool)
