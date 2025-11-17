from conduit.tools.tool import ToolCall, ToolCallError
from conduit.tools.tool import Tool
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path


class FileSearchParameters(BaseModel):
    path: str = Field(description="Absolute file path")
    query_string: str = Field(description="Text to search for")


class FileSearchToolCall(ToolCall):
    """Search a file for lines matching a query string."""

    tool_name: Literal["file_search"]
    parameters: FileSearchParameters


def file_search(call: FileSearchToolCall) -> str:
    """
    Search a file for lines containing the query string.

    Args:
        call: The tool call containing path and query string.

    Returns:
        String with matching lines (with line numbers).
    """
    file_path = Path(call.parameters.path)

    if not file_path.exists():
        raise ToolCallError(f"File does not exist: {call.parameters.path}")

    if not file_path.is_file():
        raise ToolCallError(f"Path is not a file: {call.parameters.path}")

    query = call.parameters.query_string.lower()

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        matches = []
        for line_num, line in enumerate(lines, start=1):
            if query in line.lower():
                matches.append(f"{line_num}: {line.rstrip()}")

        if not matches:
            return f"No matches found for '{call.parameters.query_string}'"

        return "\n".join(matches)

    except UnicodeDecodeError:
        raise ToolCallError("File is not readable as text")


FileSearchTool = Tool(tool_call_schema=FileSearchToolCall, function=file_search)
