from conduit.tools.tool import ToolCall, ToolCallError
from conduit.tools.tool import Tool
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path


class ListFilesParameters(BaseModel):
    path: str = Field(description="Directory path to list")


class ListFilesToolCall(ToolCall):
    """List all files and directories in a given path."""

    tool_name: Literal["list_files"]
    parameters: ListFilesParameters


def list_files(call: ListFilesToolCall) -> str:
    """
    List all files and directories in a path.

    Args:
        call: The tool call containing the directory path.

    Returns:
        String listing of files and directories, one per line.
    """
    dir_path = Path(call.parameters.path)

    if not dir_path.exists():
        raise ToolCallError(f"Path does not exist: {call.parameters.path}")

    if not dir_path.is_dir():
        raise ToolCallError(f"Path is not a directory: {call.parameters.path}")

    try:
        items = sorted(dir_path.iterdir())
        result = []
        for item in items:
            prefix = "[DIR] " if item.is_dir() else "[FILE]"
            result.append(f"{prefix} {item.name}")
        return "\n".join(result)
    except PermissionError:
        raise ToolCallError(f"Permission denied: {call.parameters.path}")


ListFilesTool = Tool(tool_call_schema=ListFilesToolCall, function=list_files)
