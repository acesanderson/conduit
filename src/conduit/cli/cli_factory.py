"""
Customize ConduitCLI quickly with a query function, if you don't worry about custom parameters.
A simple query_function that ONLY takes query_input and returns a Response object is enough.
"""

from conduit.cli.cli_class import (
    ConduitCLI,
    DEFAULT_PREFERRED_MODEL,
    DEFAULT_VERBOSITY,
)
from conduit.cli.query_function import QueryFunctionProtocol
from typing import TYPE_CHECKING
import inspect

if TYPE_CHECKING:
    from conduit.sync import Verbosity

"""
class QueryFunctionProtocol(Protocol):
    def __call__(
        self,
        inputs: dict[str, str],
        preferred_model: str,
        include_history: bool,
        verbose: Verbosity = Verbosity.PROGRESS,
    ) -> Response: ...
"""


def validate_query_function_signature(query_function) -> bool:
    """
    Validate that the query_function has the expected simple signature.
    It should only take a single parameter: query_input (str).
    """

    sig = inspect.signature(query_function)
    params = sig.parameters

    # Check that there is exactly one parameter
    if len(params) != 1:
        return False

    # Check that the parameter is named 'query_input' and is of type str
    param = next(iter(params.values()))
    if param.name != "query_input":
        return False

    # Optionally, check for type annotation (if present)
    if param.annotation not in (str, inspect._empty):
        return False

    return True


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


def cli_factory(query_function: QueryFunctionProtocol) -> ConduitCLI:
    """
    Factory function to create a ConduitCLI instance with a simple query function.
    """
    if not validate_query_function_signature(query_function):
        raise ValueError(
            "The provided query_function does not match the expected signature. "
            "It should only take a single parameter: query_input (str)."
        )
    wrapped_function = wrap_query_function(query_function)
    cli_instance = ConduitCLI(query_function=wrapped_function)
    return cli_instance
