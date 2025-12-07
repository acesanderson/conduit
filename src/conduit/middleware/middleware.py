from __future__ import annotations
from conduit.middleware.context_manager import (
    middleware_context_manager_sync,
    middleware_context_manager_async,
)
from conduit.domain.request.request import Request
import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.result.result import ConduitResult
    from conduit.middleware.protocol import Instrumentable
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_params(*args, **kwargs) -> tuple[Instrumentable, Request]:
    """Common logic to extract 'self' and 'request' from args and kwargs."""
    request: Request
    if len(args) > 1 and isinstance(args[1], Request):
        self = args[0]
        request = args[1]
    elif "request" in kwargs and isinstance(kwargs["request"], Request):
        self = args[0]
        request = kwargs["request"]
    else:
        raise ValueError(
            "The decorated function must have a Request as its second positional argument or as a 'request' keyword argument."
        )
    return self, request


def middleware_sync(
    func: Callable[[Request], ConduitResult],
) -> Callable[[Request], ConduitResult]:
    @functools.wraps(func)
    def sync_wrapper(*args: object, **kwargs: object) -> ConduitResult:
        self, request = get_params(*args, **kwargs)

        # All the magic is in here
        with middleware_context_manager_sync(self, request) as ctx:
            # Check for cache hit (passed back from context manager)
            if ctx["cache_hit"] is True:
                result = ctx["result"]
            # If no cache hit, execute the function
            else:
                result = func(*args, **kwargs)
                # Pass result back to context manager for post-processing
                ctx["result"] = result

        assert "result" in ctx, (
            "Something went wrong in the middleware context manager; no result found."
        )

        return result

    return sync_wrapper


def middleware_async(
    func: Callable[[Request], ConduitResult],
) -> Callable[[Request], ConduitResult]:
    @functools.wraps(func)
    async def async_wrapper(*args: object, **kwargs: object) -> ConduitResult:
        self, request = get_params(*args, **kwargs)

        # All the magic is in here
        with middleware_context_manager_async(self, request) as ctx:
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
