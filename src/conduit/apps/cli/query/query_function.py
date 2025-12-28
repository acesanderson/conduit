"""
MAJOR REFACTOR INCOMING:
- should this use Model or Conduit?
- handle persistence?
Our default query function -- passed into ConduitCLI as a dependency.
Define your own for customization.
"""

from conduit.config import settings
from conduit.core.prompt.prompt import Prompt
from conduit.domain.result.response import GenerationResponse
from conduit.utils.progress.verbosity import Verbosity
from conduit.apps.cli.utils.printer import Printer
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import logging

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


# Our protocol
@runtime_checkable
class CLIQueryFunctionProtocol(Protocol):
    """
    Protocol for a query function. Customized query functions should match this signature.
    """

    def __call__(
        self,
        inputs: CLIQueryFunctionInputs,
    ) -> GenerationResponse: ...


# Now, our default implementation -- the beauty of LLMs with POSIX philosophy
def default_query_function(
    inputs: CLIQueryFunctionInputs,
) -> GenerationResponse:
    """
    Default query function.
    """
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
    )
    logger.info(f"Using model: {preferred_model}")
    response = conduit()
    return response
