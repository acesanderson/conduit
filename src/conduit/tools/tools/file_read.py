from conduit.tools.tool import ToolCall
from pydantic import BaseModel, Field
from typing import Literal


class FileReadParameters(BaseModel):
    path: str = Field(description="Absolute file path")


class FileReadToolCall(ToolCall):
    """Read a file's contents."""

    tool_name: Literal["file_read"]
    parameters: FileReadParameters
