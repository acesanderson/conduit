from conduit.tools.tool import Tool, ToolCall
from conduit.tools.tools import FetchUrlTool
from conduit.tools.registry import ToolRegistry
from conduit.parser.stream.parsers import XMLStreamParser
from conduit.model.model import Model
from conduit.prompt.prompt import Prompt
from pathlib import Path

SYSTEM_PROMPT_PATH = (
    Path(__file__).parent.parent / "tools" / "prompts" / "system_prompt_template.jinja2"
)
PREFERRED_MODEL = "gpt"

registry = ToolRegistry()
registry.register(FetchUrlTool)
model = Model(PREFERRED_MODEL)


def generate_system_prompt(registry: ToolRegistry) -> str:
    system_prompt_template = SYSTEM_PROMPT_PATH.read_text()
    # Input variables
    system_prompt = ""
    tools_schema = ""


"""
from conduit.tools.registry import ToolRegistry
from conduit.tools.tools import FileReadTool

# Initialize and register tools
registry = ToolRegistry()
registry.register(FileReadTool)

# Generate schema for system prompt
system_prompt_context = registry.xml_schema

# Execute a tool call (typically parsed from LLM output)
result = registry.parse_and_execute(
    tool_name="file_read",
    parameters={"path": "/path/to/file.txt"}
)
"""


"""
parser = XMLStreamParser(stream, tag_name="function_calls")

# Parses the stream, stopping and closing connection once the full XML tag is found
text_content, xml_object, full_buffer = parser.parse(close_on_match=True)

if xml_object:
    print(f"Extracted XML: {xml_object}")
"""
