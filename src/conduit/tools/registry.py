from conduit.tools.tool import ToolCall, Tool
from pathlib import Path

TOOLS_DIR = Path(__file__).parent / "tools"


class ToolRegistry:
    def __init__(self):
        self._tools: list[Tool] = []
        self._tool_map: dict[str, Tool] = {}  # Fast lookup

    @property
    def tools(self) -> list[Tool]:
        """Return a list of all registered tools."""
        return self._tools

    @property
    def xml_schema(self) -> str:
        """
        The XML schema for all registered tools, for LLM system prompt.
        """
        return "\n".join(tool.xml_schema for tool in self._tools)

    @property
    def json_schema(self) -> str:
        """
        The JSON schema for all registered tools, for LLM system prompt.
        """
        return "\n".join(tool.json_schema for tool in self._tools)

    def register(self, tool: Tool) -> None:
        """
        Register a callable executor function for a given tool name.
        """
        self._tools.append(tool)
        self._tool_map[tool.name] = tool

    def parse_and_execute(self, tool_name: str, parameters: dict) -> str:
        """
        Parse raw LLM output into a validated ToolCall, then execute it.

        This is your main entry point from the stream parser.
        """
        if tool_name not in self._tool_map:
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        tool = self._tool_map[tool_name]

        # Validate using the specific ToolCall class
        validated_call = tool.tool_call_schema.from_xml(tool_name, parameters)

        # Execute with the validated call
        return tool.execute(validated_call)

    def execute(self, call: ToolCall) -> str:
        """Execute a tool call by looking up and invoking its registered handler function."""
        for tool in self._tools:
            if tool.name == call.tool_name:
                return tool.execute(call)

        raise ValueError(f"Tool '{call.tool_name}' is not registered.")


if __name__ == "__main__":
    from conduit.tools.tools.file_read import FileReadTool

    # Create registry and register a tool
    registry = ToolRegistry()
    registry.register("file_read", file_read)

    # Execute a tool call
    params = FileReadParameters(path=str(Path(__file__)))
    tool_call = FileReadToolCall(tool_name="file_read", parameters=params)
    # print(tool_call.model_dump_json(indent=2))
    # result = registry.execute(tool_call)
    # print(f"Result: {result}")
