"""
MAJOR REFACTOR INCOMING:
- should this use Model or Conduit?
- handle persistence?
Our default query function -- passed into ConduitCLI as a dependency.
Define your own for customization.
"""

from __future__ import annotations

from conduit.config import settings
from conduit.core.prompt.prompt import Prompt
from conduit.utils.progress.verbosity import Verbosity
from conduit.apps.cli.utils.printer import Printer
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import logging

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation

logger = logging.getLogger(__name__)


@dataclass
class CLIQueryFunctionInputs:
    """
    This has defaults, besides query_input, which is required.
    All query functions need to accept this object.
    Best to handle all possible inputs, so that CLI acts as user expects.
    If not using a particular input, consider logging a warning to inform the user.
    """

    # Project
    project_name: str
    # Prompt inputs
    query_input: str
    printer: Printer
    context: str = ""
    append: str = ""
    system_message: str = ""
    # Configs
    temperature: float | None = None
    cache: bool = True
    local: bool = False
    preferred_model: str = settings.preferred_model
    verbose: Verbosity = Verbosity.PROGRESS
    include_history: bool = True  # whether to include conversation history in messages
    ephemeral: bool = False  # whether to avoid persisting this conversation
    search: bool = False  # Enable web search + URL fetch tools
    client_params: dict = field(default_factory=dict)
    image_path: str | None = None


# Our protocol
@runtime_checkable
class CLIQueryFunctionProtocol(Protocol):
    """
    Protocol for a query function. Customized query functions should match this signature.
    """

    def __call__(
        self,
        inputs: CLIQueryFunctionInputs,
    ) -> Conversation: ...


def _search_query_function(inputs: CLIQueryFunctionInputs) -> Conversation:
    """
    Query function variant that registers web_search and fetch_url as tools,
    enabling the Engine's multi-turn GENERATE → EXECUTE → TERMINATE loop.
    """
    from conduit.capabilities.tools.registry import ToolRegistry
    from conduit.capabilities.tools.tools.fetch.fetch import fetch_url, web_search
    from conduit.core.conduit.conduit_sync import ConduitSync
    from conduit.domain.request.generation_params import GenerationParams

    tool_registry = ToolRegistry()
    tool_registry.register_function(web_search)
    tool_registry.register_function(fetch_url)

    combined_query = "\n\n".join(
        [inputs.query_input, inputs.context, inputs.append]
    ).strip()
    prompt = Prompt(combined_query)

    params = GenerationParams(
        model=inputs.preferred_model,
        system=inputs.system_message or None,
        temperature=inputs.temperature,
    )

    options = settings.default_conduit_options()
    opt_updates: dict = {
        "verbosity": inputs.verbose,
        "tool_registry": tool_registry,
        "include_history": inputs.include_history,
    }

    if inputs.local:
        opt_updates["use_remote"] = True  # --local routes through HeadwaterServer (remote from API perspective)

    if inputs.cache:
        cache_name = inputs.project_name or settings.default_project_name
        opt_updates["cache"] = settings.default_cache(project_name=cache_name)

    if not inputs.ephemeral:
        repo_name = inputs.project_name or settings.default_project_name
        opt_updates["repository"] = settings.default_repository(project_name=repo_name)

    options = options.model_copy(update=opt_updates)

    conduit = ConduitSync(prompt=prompt, params=params, options=options)
    return conduit.run()


# Now, our default implementation -- the beauty of LLMs with POSIX philosophy
def default_query_function(
    inputs: CLIQueryFunctionInputs,
) -> Conversation:
    """
    Default query function.
    """
    if inputs.search:
        return _search_query_function(inputs)
    logger.debug("Running default_query_function...")
    # Extract inputs from dict
    project_name = inputs.project_name
    query_input: str = inputs.query_input
    context: str = inputs.context
    append: str = inputs.append
    local: bool = inputs.local
    preferred_model: str = inputs.preferred_model
    verbose: Verbosity = inputs.verbose
    include_history: bool = inputs.include_history
    ephemeral: bool = False
    cache = inputs.cache
    system = inputs.system_message

    # ConduitCLI's default POSIX philosophy: embrace pipes and redirection
    combined_query = "\n\n".join([query_input, context, append]).strip()
    logger.info("Combined query prepared.")

    from conduit.core.conduit.conduit_sync import ConduitSync

    client_params = inputs.client_params or None  # normalize empty dict -> None

    prompt = Prompt(combined_query)
    conduit = ConduitSync.create(
        project_name=project_name,
        model=preferred_model,
        prompt=prompt,
        system=system,
        cache=cache,
        persist=not ephemeral,
        verbose=verbose,
        debug_payload=False,  # Change to True to debug payloads
        include_history=include_history,
        use_remote=local,
        client_params=client_params,
    )
    logger.info(f"Using model: {preferred_model}")
    response = conduit()
    return response
