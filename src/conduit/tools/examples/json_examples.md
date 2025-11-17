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
