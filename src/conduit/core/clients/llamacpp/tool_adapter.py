from __future__ import annotations
from conduit.core.clients.openai.tool_adapter import convert_tool_to_openai
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conduit.capabilities.tools.tool import Tool


def convert_tool_to_llamacpp(tool: Tool) -> dict[str, Any]:
    return convert_tool_to_openai(tool)
