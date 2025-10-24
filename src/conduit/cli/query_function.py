"""
Our default query function -- passed into ConduitCLI as a dependency.
Define your own for customization.
"""

from conduit.sync import Prompt, Model, Conduit, Response, Verbosity
from conduit.model.models.modelstore import ModelStore
from typing import Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


# First, our protocol
@runtime_checkable
class QueryFunctionProtocol(Protocol):
    """
    Protocol for a query function. Customized query functions should match this signature.
    """

    def __call__(
        self,
        inputs: dict[str, str],
        preferred_model: str,
        include_history: bool,
        verbose: Verbosity = Verbosity.PROGRESS,
    ) -> Response: ...


# Now, our default implementation -- the beauty of LLMs with POSIX philosophy
def default_query_function(
    inputs: dict[str, str],
    preferred_model: str,
    include_history: bool,
    verbose: Verbosity = Verbosity.PROGRESS,
) -> Response:
    """
    Default query function.
    Note that the input dict will contain keys for all possible inputs, from the positional query, piped in context, and a potential "append" string.
    The simplest query functions will likely only want query_input.
    """
    logger.debug("Running default_query_function...")
    # Extract inputs from dict
    query_input: str = inputs.get("query_input", "")
    context: str = (
        "<context>\n" + inputs.get("context", "") + "\n</context>"
        if inputs.get("context")
        else ""
    )
    append: str = inputs.get("append", "")
    local: bool = inputs.get("local", False)

    # ConduitCLI's default POSIX philosophy: embrace pipes and redirection
    combined_query = "\n\n".join([query_input, context, append])

    # Inject system message if provided and message store exists
    if Conduit.message_store:
        system_message = inputs.get("system_message", "")
        if system_message:
            Conduit.message_store.ensure_system_message(system_message)

    # Our chain
    if local:
        from conduit.model.remote_model import RemoteModel

        logger.info("Using local model.")
        if preferred_model not in ModelStore().local_models():
            preferred_model = "gpt-oss:latest"
        logger.info(f"Using model: {preferred_model}")
        model = RemoteModel(preferred_model)
        prompt_str = combined_query
        response = model.query(query_input=prompt_str, verbose=verbose)
    else:
        logger.info("Using remote model.")
        model = Model(preferred_model)
        logger.info(f"Using model: {preferred_model}")
        prompt = Prompt(combined_query)
        conduit = Conduit(prompt=prompt, model=model)
        response = conduit.run(verbose=verbose, include_history=include_history)
    assert isinstance(response, Response), "Response is not of type Response"
    return response
