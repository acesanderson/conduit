from conduit.tools.tool import ToolCall, Tool
from pathlib import Path
import logging
import os

# Set up logging
log_level = int(os.getenv("PYTHON_LOG_LEVEL", "2"))  # Default to INFO
levels = {1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
logging.basicConfig(
    level=levels.get(log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).parent / "tools"


class ToolRegistry:
    def __init__(self):
        logger.info("Initializing ToolRegistry")
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
        logger.debug(f"Registering tool: {tool.name}")
        self._tools.append(tool)
        self._tool_map[tool.name] = tool

    def register_all(self) -> None:
        """
        Register all tools found in the tools directory.
        NOTE: this will be get very large as more tools are added; consider specifically only
        loading the tools you need.
        """
        logger.info("Registering all tools from conduit.tools.tools")
        from conduit.tools.tools import AllTools

        for tool in AllTools:
            self.register(tool)

    def parse_and_execute(self, tool_name: str, parameters: dict) -> str:
        """
        Parse raw LLM output into a validated ToolCall, then execute it.

        This is your main entry point from the stream parser.
        """
        logger.info(f"Parsing and executing tool call for tool: {tool_name}")
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
