"""
We will have async by default; we can also create a Sync (for simple one-off queries) with:
```python
async def one_off_query(): ...

asyncio.run(one_off_query())
```
"""

from conduit.config import settings
from conduit.capabilities.tools.tools.fetch_url import FetchUrlTool
from conduit.capabilities.tools.parsing_implementation import is_tool_call, execute_tool_call
from conduit.capabilities.tools.registry import ToolRegistry
from conduit.core.parser.stream.parsers import XMLStreamParser
from conduit.domain.message.textmessage import TextMessage
from conduit.core.model.model_async import ModelAsync
from conduit.core.prompt.prompt import Prompt
from pathlib import Path
import asyncio

SYSTEM_PROMPT_PATH = (
    Path(__file__).parent.parent / "tools" / "prompts" / "system_prompt_template.jinja2"
)
PREFERRED_MODEL = settings.preferred_model


def generate_system_prompt(registry: ToolRegistry) -> str:
    if len(registry.tools) == 0:
        raise ValueError("No tools registered in the registry.")
    system_prompt_template = SYSTEM_PROMPT_PATH.read_text()
    # Input variables
    input_variables = {
        "system_prompt": settings.system_prompt,
        "tools_schema": registry.xml_schema,
        "example_call": registry.tools[0].xml_example,
    }
    system_prompt = Prompt(system_prompt_template)
    rendered = system_prompt.render(input_variables=input_variables)
    return rendered


async def main():
    registry = ToolRegistry()
    registry.register(FetchUrlTool)
    model = ModelAsync(PREFERRED_MODEL)
    system_prompt = generate_system_prompt(registry)
    system_message = TextMessage(role="system", content=system_prompt)
    prompt_str = "Summarize this article: https://news.ucsc.edu/2025/11/sharf-preconfigured-brain/"
    user_message = TextMessage(role="user", content=prompt_str)
    messages = [system_message, user_message]
    stream = await model.query(query_input=messages, stream=True)
    parser = XMLStreamParser(stream, tag_name="tool_call")
    text_content, xml_object, full_buffer = await parser.parse_async(
        close_on_match=True
    )
    if xml_object:
        print(f"Extracted XML: {xml_object}")
    print(f"Remaining Text: {text_content}")
    # Async

    if is_tool_call(xml_object):
        print(f"Is tool call: {True}")
        # Execute the tool call (Note: this is async method)
        result = await execute_tool_call(xml_object, registry)
        print(f"Tool Execution Result: {result}")


asyncio.run(main())


"""
# Execute a tool call (typically parsed from LLM output)
result = registry.parse_and_execute(
    tool_name="file_read",
    parameters={"path": "/path/to/file.txt"}
)

# Parses the stream, stopping and closing connection once the full XML tag is found
text_content, xml_object, full_buffer = parser.parse(close_on_match=True)
"""
