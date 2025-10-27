"""
Customize ConduitCLI quickly with a query function, if you don't worry about custom parameters.
A simple query_function that ONLY takes query_input and returns a Response object is enough.
"""

from conduit.cli.cli_class import (
    ConduitCLI,
    DEFAULT_PREFERRED_MODEL,
    DEFAULT_VERBOSITY,
)
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from conduit.sync import Verbosity


def wrap_query_function(query_function):
    """
    Wrap a simple query function to match the expected signature for ConduitCLI.

    Discard extra parameters, as well as extra context (i.e. contex / append).
    """

    def wrapped(
        inputs: dict[str, str],
        preferred_model: str = DEFAULT_PREFERRED_MODEL,
        verbose: "Verbosity" = DEFAULT_VERBOSITY,
        nopersist: bool = False,
    ):
        query_input = inputs.get("query_input", "")
        _ = preferred_model  # Discarded
        _ = verbose  # Discarded
        _ = nopersist  # Discarded
        return query_function(query_input)

    return wrapped

def validate_protocol(query_function: Callable)

def cli_factory(query_function) -> ConduitCLI:
    """
    Factory function to create a ConduitCLI instance with a simple query function.
    """
    wrapped_function = wrap_query_function(query_function)
    cli_instance = ConduitCLI(query_function=wrapped_function)
    return cli_instance
