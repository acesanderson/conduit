from __future__ import annotations
from conduit.capabilities.tools.tool import Tool, ToolCallError
from conduit.capabilities.tools.tool_function import ToolFunction
from conduit.domain.message.message import ToolCall
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.capabilities.skills.registry import SkillRegistry


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._skill_registry: SkillRegistry = None

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool with name '{name}' is not registered.")
        return self._tools[name]

    async def call_tool(self, tool_call: ToolCall) -> str:
        tool = self.get_tool(tool_call.function_name)

        # Create a copy of arguments to run the function so we don't pollute
        # the message history with non-serializable objects (like registries).
        func_args = tool_call.arguments.copy()

        # Inject registries if calling the ensure_skill tool
        if tool.name == "enable_skill" and self._skill_registry is not None:
            func_args["_skill_registry"] = self._skill_registry
            func_args["_tool_registry"] = self

        try:
            result = await tool.func(**func_args)
        except Exception as e:
            raise ToolCallError(f"Error calling tool '{tool.name}': {e}") from e
        return result if isinstance(result, str) else str(result)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def register_function(self, func: ToolFunction) -> None:
        tool = Tool.from_function(func)
        self.register(tool)

    def register_functions(self, funcs: list[ToolFunction]) -> None:
        for func in funcs:
            self.register_function(func)

    def enable_skills(self, skill_registry: SkillRegistry):
        """
        Injects skill-enabling tools into the registry, as well as adding the ensure_skill tool.
        """
        from conduit.capabilities.skills.tool import enable_skill_tool

        self._skill_registry = skill_registry
        self.register(enable_skill_tool)
