from conduit.tools.tool_call import ToolCall
from collections.abc import Callable


class ToolRegistry:
    """
    Registry for managing and executing tool functions by name.

    Container that maintains a mapping of tool names to their executor functions.
    Enables registration of callable handlers and execution of tool calls by
    looking up and invoking the corresponding function with validated parameters.

    Attributes:
        _executors: Dictionary mapping tool names to callable executor functions.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register("file_read", lambda path: open(path).read())
        >>> tool_call = ToolCall(tool_name="file_read", parameters={"path": "/tmp/test.txt"})
        >>> result = registry.execute(tool_call)
    """

    def __init__(self):
        self._executors: dict[str, Callable[..., str]] = {}

    def register(self, tool_name: str, func: Callable[..., str]) -> None:
        """
        Register a callable executor function for a given tool name.
        """
        self._executors[tool_name] = func

    def execute(self, call: ToolCall) -> str:
        """Execute a tool call by looking up and invoking its registered handler function."""
        func = self[call.tool_name]  # Uses __getitem__
        return func(**call.parameters.model_dump())

    def __getitem__(self, tool_name: str) -> Callable[..., str]:
        """Get the executor function for a given tool name using [] syntax."""
        if tool_name not in self._executors:
            raise KeyError(f"Tool '{tool_name}' is not registered")
        return self._executors[tool_name]
