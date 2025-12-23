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

    # Prompt inputs
    query_input: str
    printer: Printer
    context: str = ""
    append: str = ""
    system_message: str = ""
    # Configs
    temperature: float = 0.7
    name: str = "default"
    cache: bool = True
    local: bool = False
    preferred_model: str = settings.preferred_model
    verbose: Verbosity = Verbosity.PROGRESS
    include_history: bool = True


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
    name: str = inputs.name
    query_input: str = inputs.query_input
    context: str = inputs.context
    append: str = inputs.append
    local: bool = inputs.local
    preferred_model: str = inputs.preferred_model
    verbose: Verbosity = inputs.verbose
    include_history: bool = inputs.include_history
    cache = inputs.cache
    system = inputs.system_message

    # ConduitCLI's default POSIX philosophy: embrace pipes and redirection
    combined_query = "\n\n".join([query_input, context, append]).strip()

    # Our chain
    if local:
        raise NotImplementedError("Local model querying is not implemented yet.")
    else:
        logger.info("Using cloud model.")
        from conduit.core.conduit.conduit_sync import ConduitSync

        prompt = Prompt(combined_query)
        if cache:
            cache_string: str | None = name
        else:
            cache_string = None
        conduit = ConduitSync.create(
            model=preferred_model,
            prompt=prompt,
            system_message=system,
            cache=cache_string,
            include_history=include_history,
            verbose=verbose,
        )
        logger.info(f"Using model: {preferred_model}")
        response = conduit()
    return response
