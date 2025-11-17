from typing import get_args
from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_name: str  # Overridden with Literal in subclasses
    parameters: BaseModel  # Overridden with a BaseModel subclass in subclasses

    def __init__(self, **data):
        """
        Initialize a ToolCall instance, enforcing subclass usage only.
        """
        if self.__class__ is ToolCall:
            raise TypeError("Cannot instantiate ToolCall directly. Use a subclass.")
        super().__init__(**data)

    @classmethod
    def to_xml_schema(cls) -> str:
        """Generate tool definition XML for system prompt."""
        # Get the tool name from the Literal type
        tool_name_field = cls.model_fields["tool_name"]
        tool_name = get_args(tool_name_field.annotation)[0]

        # Get description from docstring
        description = cls.__doc__.strip() if cls.__doc__ else ""

        # Get parameter schema
        params_field = cls.model_fields["parameters"]
        params_model = params_field.annotation
        schema = params_model.model_json_schema()

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        params_xml = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            is_required = "true" if param_name in required else "false"

            params_xml.append(
                f'    <parameter name="{param_name}" type="{param_type}" '
                f'required="{is_required}">{param_desc}</parameter>'
            )

        params_str = "\n".join(params_xml)

        return f"""<tool name="{tool_name}">
  <description>{description}</description>
  <parameters>
{params_str}
  </parameters>
</tool>"""

    @classmethod
    def from_xml(cls, tool_name: str, parameters: dict):
        """Parse and validate LLM's XML tool call."""
        return cls.model_validate({"tool_name": tool_name, "parameters": parameters})


"""
Example:

class FileReadParameters(BaseModel):
    path: str = Field(description="Absolute file path")


class FileReadToolCall(ToolCall):
    \"\"\"Read a file's contents.\"\"\"

    tool_name: Literal["file_read"]
    parameters: FileReadParameters
"""
