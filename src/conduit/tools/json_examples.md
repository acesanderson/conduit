# Example of a function schema
{
    "type": "function",
    "function": {
        "name": "file_read",
        "description": "Reads the content of a specified file path. Use this to access skill definitions.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute file path to the skill's .md file.",
                }
            },
            "required": ["path"],
        },
    },
}

# Example of a tool call using the function schema
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123xyz",
      "type": "function",
      "function": {
        "name": "file_read",
        "arguments": "{\"path\": \"/path/to/my_skill/SKILL.md\"}"
      }
    }
  ]
}

# Example of parallel tool calls using the function schema
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_read_skill_abc",
      "type": "function",
      "function": {
        "name": "file_read",
        "arguments": "{\"path\": \"/skills/python/SKILL.md\"}"
      }
    },
    {
      "id": "call_list_files_xyz",
      "type": "function",
      "function": {
        "name": "list_files",
        "arguments": "{\"path\": \"/skills/python/references/\"}"
      }
    }
  ]
}
