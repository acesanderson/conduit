from conduit.capabilities.tools.tool import Tool, ToolCallError
from conduit.domain.message.message import ToolCall


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool with name '{name}' is not registered.")
        return self._tools[name]

    def call_tool(self, tool_call: ToolCall) -> str:
        tool = self.get_tool(tool_call.function_name)
        try:
            result = tool.func(**tool_call.arguments)
        except Exception as e:
            raise ToolCallError(f"Error calling tool '{tool.name}': {e}") from e
        return result if isinstance(result, str) else str(result)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())
