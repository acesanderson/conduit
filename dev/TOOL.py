from conduit.capabilities.tools.registry import ToolRegistry
from conduit.capabilities.tools.tools.fetch_url import fetch_url
from conduit.capabilities.tools.tools.file_read import file_read
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.sync import Model, Conduit, Prompt, Verbosity
from typing import Annotated


# async def get_weather(
#     location: Annotated[str, "The location to get the weather for"],
# ) -> dict[str, str]:
#     """
#     Get the current weather for a given location.
#     """
#     # For demonstration purposes, we'll return a mock weather report.
#     return {
#         "location": location,
#         "temperature": "72°F",
#         "condition": "Partly Cloudy",
#     }
#
#
# async def get_timezone(
#     location: Annotated[str, "The location to get the timezone for"],
# ) -> dict[str, str]:
#     """
#     Get the current timezone for a given location.
#     """
#     # For demonstration purposes, we'll return a mock timezone.
#     return {
#         "location": location,
#         "timezone": "Eastern Standard Time (EST)",
#         "utc_offset": "-5:00",
#     }
#
#
# async def buy_groceries(
#     items: Annotated[list[str], "List of grocery items to buy"],
#     store: Annotated[str, "The store to buy groceries from"],
# ) -> dict[str, str]:
#     """
#     Buy groceries from a specified store.
#     """
#     # For demonstration purposes, we'll return a mock purchase confirmation.
#     return {
#         "store": store,
#         "items_purchased": ", ".join(items),
#         "status": "Purchase successful",
#     }


registry = ToolRegistry()
# registry.register_functions([get_weather, get_timezone, buy_groceries])
registry.register_functions([fetch_url, file_read])
options = ConduitOptions(
    project_name="test", tool_registry=registry, verbosity=Verbosity.COMPLETE
)
params = GenerationParams(model="gpt-4o")
# params = GenerationParams(model="flash")
# params = GenerationParams(model="llama3.1:latest")
# params = GenerationParams(model="haiku")

# m = Model(options=options, params=params)
# response = m.query("What's the weather like in New York and what time zone is it?")
# prompt = Prompt("What's the weather like in New York and what time zone is it?")
# prompt = Prompt(
#     "Summarize this article: https://en.wikipedia.org/wiki/Artificial_intelligence"
# )
prompt = Prompt("Read this file: /home/fishhouses/.mailmap")
c = Conduit(prompt=prompt, options=options, params=params)
response = c.run()
response.pretty_print()
