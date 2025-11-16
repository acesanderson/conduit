from pydantic import BaseModel, Field


class LLMToolMixin:
    @classmethod
    def _llm_schema_prompt(cls) -> str:
        """Generates an XML schema description for the LLM system prompt."""
        # Get the tool's main docstring
        tool_description = cls.__doc__ or "No description."

        # Get the class name for the root tag
        root_tag = cls.__name__

        # Build the XML example tags from fields
        xml_example_lines = []
        for field_name, field_info in cls.model_fields.items():
            # Use field description as placeholder
            description = field_info.description or f"Your {field_name} here"
            xml_example_lines.append(f"  <{field_name}>{description}</{field_name}>")

        xml_example = "\n".join(xml_example_lines)

        # Assemble the final prompt block
        return f"""
<tool>
  <name>{root_tag}</name>
  <description>{tool_description.strip()}</description>
  <format>
<{root_tag}>
{xml_example}
</{root_tag}>
  </format>
</tool>
"""


class SearchTool(LLMToolMixin, BaseModel):
    """Performs a web search for the given query."""

    query: str = Field(description="The exact search query to execute.")
    region: str | None = Field(default=None, description="ISO code for search region.")
