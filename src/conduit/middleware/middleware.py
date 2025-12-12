from __future__ import annotations
from conduit.middleware.context_manager import middleware_context_manager
from conduit.domain.request.request import GenerationRequest
import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.result.result import GenerationResult
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_request(*args, **kwargs) -> GenerationRequest:
    """
    Common logic to extract request from args and kwargs.
    """
    if len(args) > 1 and isinstance(args[1], GenerationRequest):
        request = args[1]
    elif "request" in kwargs and isinstance(kwargs["request"], GenerationRequest):
        request = kwargs["request"]
    else:
        raise ValueError(
            "The decorated function must have a GenerationRequest as its second positional argument or as a 'request' keyword argument."
        )
    return request


def middleware(
    func: Callable[[GenerationRequest], GenerationResult],
) -> Callable[[GenerationRequest], GenerationResult]:
    @functools.wraps(func)
    async def async_wrapper(*args: object, **kwargs: object) -> GenerationResult:
        request = get_request(*args, **kwargs)

        # All the magic is in here
        with middleware_context_manager(request) as ctx:
            # Check for cache hit (passed back from context manager)
            if ctx["cache_hit"] is True:
                result = ctx["result"]
            # If no cache hit, execute the function
            else:
                result = await func(*args, **kwargs)
                # Pass result back to context manager for post-processing
                ctx["result"] = result

        assert "result" in ctx, (
            "Something went wrong in the middleware context manager; no result found."
        )

        return result

    return async_wrapper
