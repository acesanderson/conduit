from conduit.tools.parsing_implementation import execute_tool_call
from conduit.tools.tool import ToolCallError
from conduit.tools.registry import ToolRegistry
from conduit.tools.tools import (
    FileReadTool,
    FileReadChunkTool,
    FileSearchTool,
    ListFilesTool,
)
from xml.etree import ElementTree as ET
from pathlib import Path


path = Path(__file__)
tool_registry = ToolRegistry()
tool_registry.register(FileReadTool)
tool_registry.register(FileReadChunkTool)
tool_registry.register(FileSearchTool)
tool_registry.register(ListFilesTool)

original_xml = f"""
<tool_call>
    <tool_name>file_read</tool_name>
    <parameters>
        <path>{str(path)}</path>
    </parameters>
</tool_call>
""".strip()
result = execute_tool_call(original_xml, tool_registry)
print(result)
