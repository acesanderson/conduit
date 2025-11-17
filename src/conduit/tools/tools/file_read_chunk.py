from conduit.tools.tool import ToolCall, ToolCallError
from conduit.tools.tool import Tool
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path


class FileReadChunkParameters(BaseModel):
    path: str = Field(description="Absolute file path")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (inclusive)")


class FileReadChunkToolCall(ToolCall):
    """Read a specific line range from a file to avoid context bloat."""

    tool_name: Literal["file_read_chunk"]
    parameters: FileReadChunkParameters


def file_read_chunk(call: FileReadChunkToolCall) -> str:
    """
    Read a specific range of lines from a file.

    Args:
        call: The tool call containing path and line range.

    Returns:
        String containing the requested lines.
    """
    file_path = Path(call.parameters.path)

    if not file_path.exists():
        raise ToolCallError(f"File does not exist: {call.parameters.path}")

    if not file_path.is_file():
        raise ToolCallError(f"Path is not a file: {call.parameters.path}")

    start = call.parameters.start_line
    end = call.parameters.end_line

    if start < 1:
        raise ToolCallError("start_line must be >= 1")

    if end < start:
        raise ToolCallError("end_line must be >= start_line")

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Convert to 0-indexed
        start_idx = start - 1
        end_idx = end

        if start_idx >= len(lines):
            raise ToolCallError(
                f"start_line {start} exceeds file length ({len(lines)} lines)"
            )

        selected_lines = lines[start_idx:end_idx]
        return "".join(selected_lines)

    except UnicodeDecodeError:
        raise ToolCallError("File is not readable as text")


FileReadChunkTool = Tool(
    tool_call_schema=FileReadChunkToolCall, function=file_read_chunk
)
