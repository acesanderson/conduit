"""
Connector between our XMLStreamingParser and the ToolRegistry.
If XMLStreamingParser has a hit:
- run is_tool_call to verify the XML structure
- run parse_tool_call to execute the tool and get the result
"""

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


def is_tool_call(xml_string: str) -> bool:
    """
    Validate whether an XML string represents a valid tool call structure.

    Checks that the XML has the expected tool call format with a root element
    named "tool_call" containing exactly two child elements: "tool_name" and
    "parameters".
    """

    root = ET.fromstring(xml_string)
    return (
        root.tag == "tool_call"
        and len(root) == 2
        and root[0].tag == "tool_name"
        and root[1].tag == "parameters"
    )


def execute_tool_call(xml_string: str, tool_registry: ToolRegistry) -> str:
    """
    Parse and execute a tool call from XML string format.

    Validates the XML structure, extracts the tool name and parameters,
    then routes execution through the ToolRegistry.
    """
    if is_tool_call(xml_string):
        try:
            root = ET.fromstring(xml_string)
            tool_name = root[0].text
            parameters = {}
            for param in root[1]:
                parameters[param.tag] = param.text

            result = tool_registry.parse_and_execute(tool_name, parameters)
            return result
        except ToolCallError as e:
            return f"Tool execution error: {str(e)}"
    else:
        raise ValueError(f"Invalid tool call XML format: {xml_string}")
