from typing import get_args, Protocol, Any
from pydantic import BaseModel, ConfigDict, Field
from collections.abc import Callable
import json


class ToolCallError(Exception):
    """Custom exception for ToolCall errors."""

    pass


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


class ToolFunction(Protocol):
    """
    Tool functions need to return a string and accept a single ToolCall argument.
    """

    def __call__(self, tool_call: ToolCall) -> str: ...


class Tool(BaseModel):
    """Tool registry object."""

    tool_call_schema: type[ToolCall]
    function: Callable[[ToolCall], str] = Field(
        description="The function to execute the tool"
    )
    # These are for dynamic example generation for system prompt
    example_query: str = Field(
        description="Example user prompt that triggers this tool"
    )
    example_params: dict[str, Any] = Field(
        description="Example parameters for the tool call"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        """Get the tool's name."""
        tool_name_field = self.tool_call_schema.model_fields["tool_name"]
        return get_args(tool_name_field.annotation)[0]

    @property
    def xml_schema(self) -> str:
        """Get the tool's XML schema."""
        return self.tool_call_schema.to_xml_schema()

    @property
    def json_schema(self) -> str:
        """Get the tool's JSON schema."""
        schema_dict: dict = self.tool_call_schema.model_json_schema()
        return json.dumps(schema_dict, indent=2)

    @property
    def xml_example(self) -> str:
        """Generate a full XML example for the system prompt."""
        # Convert dict params to XML tags
        params_xml = "\n".join(
            f"    <{k}>{v}</{k}>" for k, v in self.example_params.items()
        )

        return f"""User: "{self.example_query}"
Assistant:
<thought>
The user has requested an action that requires the '{self.name}' tool. I will execute it with the appropriate parameters.
</thought>
<tool_call>
    <tool_name>{self.name}</tool_name>
    <parameters>
{params_xml}
    </parameters>
</tool_call>"""

    def execute(self, call: ToolCall) -> str:
        """Execute this tool with a validated call."""
        return self.function(call)
