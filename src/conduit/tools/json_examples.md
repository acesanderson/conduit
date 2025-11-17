# Function schema for a tool that reads a file from a given path
{
  "type": "object",
  "properties": {
    "tool_name": {
      "type": "string",
      "enum": ["file_read"]
    },
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string"
        }
      },
      "required": ["path"],
      "additionalProperties": false
    }
  },
  "required": ["tool_name", "parameters"],
  "additionalProperties": false
}


# Example of a tool call using the function schema
{
    "tool_name": "file_read",
    "parameters": {
        "path": "/skills/docx/SKILL.md"
    }
}

# pydantic
from pydantic import BaseModel, Field

class ToolCall(BaseModel):
    """No additional properties are allowed, this is for methods."""

    def to_xml_schema(self) -> str:
        """Convert the pydantic model to an XML schema representation."""
        ...

    def __call__(self, *args, **kwargs) -> str:
        return self.function(**self.parameters.dict())




class FileReadParameters(BaseModel):
    path: str = Field(
        ...,
        description="The path to the file to be read.",
    )

class FileReadToolCall(BaseModel):
    """
    Request a file by its path.
    """
    tool_name: Literal["file_read"]
    parameters: FileReadParameters

    # The function to run (not part of the schema)
    function: Callable[..., str] = Field(
        default=my_custom_file_read_function,
        exclude=True,
    )


