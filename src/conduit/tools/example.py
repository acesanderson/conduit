import json
from pydantic import BaseModel
from pydantic import Field
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Type

# --- Base Definitions ---


class BaseTool(ABC):
    """
    Abstract base class for a self-contained, runnable tool.
    It bundles the schema (for the LLM) and the execution (for the app).
    """

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel]

    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """The actual logic to execute the tool."""
        pass

    def get_json_schema(self) -> dict:
        """Generates the OpenAI-compatible JSON schema."""
        schema = self.args_schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def validate_and_run(self, llm_args: dict) -> Any:
        """Validates the LLM's arguments and then runs the tool."""
        try:
            validated_args = self.args_schema.model_validate(llm_args)
            return self._run(**validated_args.model_dump())
        except Exception as e:
            # Handle validation or runtime errors
            return f"Error running {self.name}: {e}"


# --- Tool Definitions ---


# 1. Define the parameters for the tool
class FileReadArgs(BaseModel):
    path: str = Field(..., description="The absolute file path to read.")


# 2. Define the tool itself
class FileReadTool(BaseTool):
    name = "file_read"
    description = "Reads the content of a specified file path."
    args_schema = FileReadArgs

    def _run(self, path: str) -> str:
        # This is where your actual file_read logic would go
        print(f"[Executing file_read on: {path}]")
        # f = open(path, 'r')...
        return f"Content of {path}..."


if __name__ == "__main__":
    # 1. You create an instance of your tool
    my_tool = FileReadTool()

    # 2. You generate the schema to send to the LLM
    print("--- Generating Schema for LLM ---")
    print(json.dumps(my_tool.get_json_schema(), indent=2))

    # 3. You receive a JSON blob (as a dict) from the LLM
    llm_output_args = {"path": "/skills/python/SKILL.md"}

    # 4. You validate AND run the tool with one call
    print("\n--- Validating and Running LLM Output ---")
    result = my_tool.validate_and_run(llm_output_args)
    print(f"Tool Result: {result}")
