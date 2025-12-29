from conduit.capabilities.tools.tool import Tool


def convert_tool_to_openai(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema.model_dump(
                by_alias=True, exclude_none=True
            ),
        },
    }
