from conduit.tools.tool import ToolCall, ToolCallError, Tool
from pydantic import BaseModel, Field
from typing import Literal


# Our pydantic classes
class FileReadParameters(BaseModel):
    path: str = Field(description="Absolute file path")


class FileReadToolCall(ToolCall):
    """Read a file's contents."""

    tool_name: Literal["file_read"]
    parameters: FileReadParameters


# Our function
def file_read(tool_call: FileReadToolCall) -> str:
    """Read a file's contents.

    Args:
        tool_call (FileReadToolCall): The tool call containing parameters.

    Returns:
        str: The contents of the file.
    """
    from pathlib import Path

    if not Path(tool_call.parameters.path).is_file():
        raise ToolCallError("The specified path is not a file.")
    if not Path(tool_call.parameters.path).exists():
        raise ToolCallError("The specified file does not exist.")
    with open(tool_call.parameters.path, "r") as file:
        contents = file.read()
    return contents


# Our tool registry object
FileReadTool = Tool(
    tool_call_schema=FileReadToolCall,
    function=file_read,
)
