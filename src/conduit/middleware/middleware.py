from __future__ import annotations
from collections.abc import Callable
from conduit.middleware.protocol import Instrumentable
from conduit.middleware.context_manager import middleware_context_manager
from conduit.domain.result.response import Response
from conduit.domain.request.request import Request
from conduit.domain.result.result import ConduitResult
from conduit.storage.odometer.token_event import TokenEvent
import functools
import logging
from typing import TYPE_CHECKING
from rich.console import Console

if TYPE_CHECKING:
    from conduit.storage.odometer.odometer_registry import OdometerRegistry

logger = logging.getLogger(__name__)
console = Console()


def middleware_sync(
    func: Callable[[Request], ConduitResult],
) -> Callable[[Request], ConduitResult]:
    @functools.wraps(func)
    def sync_wrapper(*args: object, **kwargs: object) -> ConduitResult:
        # 1. Validate & Extract
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

        # All the magic is in here
        with middleware_context_manager(self, request) as ctx:
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
        # 1. Validate & Extract
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

        # All the magic is in here
        async with middleware_context_manager(self, request) as ctx:
            result = await func(*args, **kwargs)
            ctx["result"] = result

        return result

    return async_wrapper
