"""
Our default query function -- passed into ConduitCLI as a dependency.
Define your own for customization.
"""

from conduit.sync import Prompt, Model, Conduit, Response, Verbosity
from conduit.model.models.modelstore import ModelStore
from pydantic import BaseModel, Field
from typing import Protocol, runtime_checkable
from rich.console import Console
import logging

logger = logging.getLogger(__name__)
console = Console()  # For pretty progress
Conduit.console = console


# Our input schema for the query function
class CLIQueryFunctionInputs(BaseModel):
    """
    This has defaults, besides query_input, which is required.
    All query functions need to accept this object.
    Best to handle all possible inputs, so that CLI acts as user expects.
    If not using a particular input, consider logging a warning to inform the user.
    """

    # Prompt inputs
    query_input: str = Field(..., description="The main query input string.")
    context: str = Field(
        default="", description="Additional context to include in the query."
    )
    append: str = Field(default="", description="String to append to the query.")
    system_message: str | None = Field(
        None, description="System message for the model."
    )
    # Configs
    name: str = Field(default="default", description="Name of the CLI app.")
    cache: bool | None = Field(
        default=True, description="Flag to indicate if caching should be used."
    )
    local: bool | None = Field(
        default=False, description="Flag to indicate if local processing is desired."
    )
    preferred_model: str = Field(
        default="haiku", description="The preferred model to use for the query."
    )
    verbose: Verbosity = Field(
        default=Verbosity.PROGRESS,
        description="Verbosity level for logging during the query.",
    )
    include_history: bool = Field(
        default=True, description="Whether to include message history in the query."
    )


# Our protocol
@runtime_checkable
class CLIQueryFunctionProtocol(Protocol):
    """
    Protocol for a query function. Customized query functions should match this signature.
    """

    def __call__(
        self,
        inputs: CLIQueryFunctionInputs,
    ) -> Response: ...


# Now, our default implementation -- the beauty of LLMs with POSIX philosophy
def default_query_function(
    inputs: CLIQueryFunctionInputs,
) -> Response:
    """
    Default query function.
    """
    logger.debug("Running default_query_function...")
    # Extract inputs from dict
    name: str = inputs.name
    query_input: str = inputs.query_input
    context: str | None = inputs.context
    append: str | None = inputs.append
    local: bool | None = inputs.local
    preferred_model: str = inputs.preferred_model
    verbose: int = inputs.verbose
    include_history: bool = inputs.include_history
    verbose = inputs.verbose
    cache = inputs.cache

    # ConduitCLI's default POSIX philosophy: embrace pipes and redirection
    combined_query = "\n\n".join([query_input, context, append])

    # Inject system message if provided and message store exists
    if Conduit.message_store:
        system_message = inputs.system_message
        if system_message:
            Conduit.message_store.ensure_system_message(system_message)

    # Our chain
    if local:
        from conduit.model.model_remote import RemoteModel

        logger.info("Using local model.")
        if preferred_model not in ModelStore().local_models():
            preferred_model = "gpt-oss:latest"
        logger.info(f"Using model: {preferred_model}")
        if cache:
            model = RemoteModel(preferred_model, cache=name)
        else:
            model = RemoteModel(preferred_model)
        prompt_str = combined_query
        response = model.query(query_input=prompt_str, verbose=verbose)
    else:
        logger.info("Using cloud model.")
        if cache:
            model = Model(preferred_model, cache=name)
        else:
            model = Model(preferred_model)
        logger.info(f"Using model: {preferred_model}")
        prompt = Prompt(combined_query)
        conduit = Conduit(prompt=prompt, model=model)
        response = conduit.run(verbose=verbose, include_history=include_history)
    assert isinstance(response, Response), "Response is not of type Response"
    return response
