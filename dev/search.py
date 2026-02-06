from conduit.config import settings
from conduit.sync import Conduit, GenerationParams, ConduitOptions, Verbosity, Prompt
from conduit.domain.conversation.conversation import Conversation
from conduit.capabilities.tools.registry import ToolRegistry
from conduit.capabilities.tools.tools.fetch.fetch import fetch_url, web_search

PROMPT_STR = """
Please summarize the latest Thoughtworks Radar.
""".strip()
# PROMPT_STR = """
# Go to Gartner Peer Insights and give me a list of every company mentioned in "Enterprise Agile Planning Tools". Return a json blob with the name of each company, the number of reviews, and the average rating.
# """.strip()
MODEL = "opus"

registry = ToolRegistry()
registry.register_functions([fetch_url, web_search])


def research_prompt(query: str) -> Conversation:
    prompt = Prompt(query)
    options = ConduitOptions(
        project_name="research",
        cache=settings.default_cache("research"),
        verbosity=Verbosity.PROGRESS,
        tool_registry=registry,
        use_cache=False,
    )
    params = GenerationParams(model=MODEL, system=settings.system_prompt)
    conduit = Conduit(prompt=prompt, options=options, params=params)
    response = conduit.run()
    return response


if __name__ == "__main__":
    conversation = research_prompt(PROMPT_STR)
    conversation.pretty_print()
