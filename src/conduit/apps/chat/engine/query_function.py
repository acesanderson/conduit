"""
Query function is injected into ConduitChat for handling queries.
Cf. the CLIQueryFunctionProtocol which we use for CLI query functions.
A ChatQueryFunction is responsible for:
- Receiving user input and context (model, message store, etc.)
- Sending the query to the model
- Handling the response and updating the message store
- Incrementing usage stats
- Returning the response for display
"""

from conduit.sync import Conduit, Response, Verbosity, Model
from conduit.domain.message.messagestore import MessageStore
from typing import Protocol, runtime_checkable
from rich.console import Console
from instructor.exceptions import InstructorRetryException
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
console = Console()  # For pretty progress
Conduit.console = console


# Our input schema for the query function
@dataclass
class ChatQueryFunctionInputs:
    user_input: str
    model: Model
    message_store: MessageStore
    verbosity: Verbosity = Verbosity.PROGRESS
    console: Console | None = None
    system_message: str | None = None


# Our protocol
@runtime_checkable
class ChatQueryFunctionProtocol(Protocol):
    """
    Protocol for a query function. Customized query functions should match this signature.
    """

    def __call__(
        self,
        inputs: ChatQueryFunctionInputs,
    ) -> Response: ...


# Now, our default implementation -- the beauty of LLMs with POSIX philosophy
def default_query_function(
    inputs: ChatQueryFunctionInputs,
):
    """
    Send a query to the model and display the response.
    """
    # Unpack inputs
    user_input = inputs.user_input
    model = inputs.model
    message_store = inputs.message_store
    console = inputs.console
    system_message = inputs.system_message
    verbosity = inputs.verbosity
    # Create model instance
    # Ignore empty queries
    if user_input.strip() == "":
        return
    # Send query to model
    try:
        # Ensure system message is set
        if system_message:
            message_store.ensure_system_message(system_message)
        # Add user message to store
        message_store.add_new(role="user", content=user_input)

        # Query model with full message history
        response = model.query(message_store.messages, verbose=verbosity)

        assert isinstance(response, Response), "Expected Response from model query"

        # Increment usage stats
        message_store.input_tokens += response.input_tokens
        message_store.output_tokens += response.output_tokens
        message_store.last_used_model_name = model.model

        # Add assistant response to store
        message_store.add_new(role="assistant", content=str(response))

        return response

    except InstructorRetryException:
        # Network failure from instructor
        return "[red]Network error. Please try again.[/red]"
