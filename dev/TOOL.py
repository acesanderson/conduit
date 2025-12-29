from conduit.capabilities.tools.tool import Tool
from conduit.capabilities.tools.registry import ToolRegistry
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.sync import Model
from typing import Annotated


def get_weather(location: Annotated[str, "The location to get the weather for"]) -> str:
    """
    Get the current weather for a given location.
    """
    # For demonstration purposes, we'll return a mock weather report.
    return f"The current weather in {location} is sunny with a temperature of 75°F."


registry = ToolRegistry()
tool = Tool.from_function(get_weather)
tool.register(registry)
options = ConduitOptions(project_name="test", tool_registry=registry)
# params = GenerationParams(model="gpt-4o")
params = GenerationParams(model="flash")

m = Model(options=options, params=params)
response = m.query("What's the weather like in New York?")
