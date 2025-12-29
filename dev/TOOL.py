from conduit.capabilities.tools.registry import ToolRegistry
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.sync import Model, Conduit, Prompt, Verbosity
from typing import Annotated


async def get_weather(
    location: Annotated[str, "The location to get the weather for"],
) -> dict[str, str]:
    """
    Get the current weather for a given location.
    """
    # For demonstration purposes, we'll return a mock weather report.
    return {
        "location": location,
        "temperature": "72°F",
        "condition": "Partly Cloudy",
    }


async def get_timezone(
    location: Annotated[str, "The location to get the timezone for"],
) -> dict[str, str]:
    """
    Get the current timezone for a given location.
    """
    # For demonstration purposes, we'll return a mock timezone.
    return {
        "location": location,
        "timezone": "Eastern Standard Time (EST)",
        "utc_offset": "-5:00",
    }


registry = ToolRegistry()
registry.register_functions([get_weather, get_timezone])
options = ConduitOptions(
    project_name="test", tool_registry=registry, verbosity=Verbosity.COMPLETE
)
params = GenerationParams(model="gpt-4o")
# params = GenerationParams(model="flash")
# params = GenerationParams(model="llama3.1:latest")
# params = GenerationParams(model="haiku")

# m = Model(options=options, params=params)
# response = m.query("What's the weather like in New York and what time zone is it?")
prompt = Prompt("What's the weather like in New York and what time zone is it?")
c = Conduit(prompt=prompt, options=options, params=params)
response = c.run()
response.pretty_print()
