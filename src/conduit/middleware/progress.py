from __future__ import annotations
from collections.abc import Callable
from conduit.middleware.protocol import Instrumentable
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


def progress_sync(
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

        # Pre execute

        with console.status(
            f"[bold gold1]Processing request for model '{request.params.model}'...",
            spinner="dots",
        ):
            result = func(*args, **kwargs)
        console.print("[bold green]Request processing complete.")

        # Post execute

        return result

    return sync_wrapper


def progress_async(
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

        # Pre execute

        with console.status(
            f"[bold gold1]Processing request for model '{request.params.model}'...",
            spinner="dots",
        ):
            result = await func(*args, **kwargs)
        console.print("[bold green]Request processing complete.")

        # Post execute

        return result

    return async_wrapper
