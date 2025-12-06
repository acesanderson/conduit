from collections.abc import Callable
from conduit.middleware.protocol import Instrumentable
from conduit.domain.result.response import Response
from conduit.domain.request.request import Request
from conduit.domain.result.result import ConduitResult
from conduit.storage.odometer.OdometerRegistry import OdometerRegistry
from conduit.storage.odometer.TokenEvent import TokenEvent
import functools
import socket
import logging

logger = logging.getLogger(__name__)


def emit_token_event(registry: OdometerRegistry) -> None:
    """
    Emit a TokenEvent to the OdometerRegistry if it exists.
    """

    # Get hostname
wn"

    event = TokenEvent(
        model=request.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def odometer_sync(
    func: Callable[[Request], ConduitResult],
) -> Callable[[Request], ConduitResult]:
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> ConduitResult:
        # 1. Validate & Extract
        self: Instrumentable
        request: Request
        if len(args) > 1 and isinstance(args[1], Request):
            self = args[0]
            request = args[1]
        elif "request" in kwargs and isinstance(kwargs["request"], Request):
            self = args[0]
            request = kwargs["request"]
        else:
            raise ValueError(
                "The decorated function must have a Request as its second positional "
                "argument or as a 'request' keyword argument."
            )
        result = func(*args, **kwargs)

        if isinstance(result, Response):
            if result.metadata.output_tokens == 0:
                # Grab registry
                registry = self.odometer_registry
                # Build TokenEvent
                model = request.model
                input_tokens = result.metadata.input_tokens
                output_tokens = result.metadata.output_tokens
                event = TokenEvent(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )




            input_tokens = result.metadata.get("input_tokens", 0)
            output_tokens = result.metadata.get("output_tokens", 0)
            registry.emit_token_event(event)

        return result

    return sync_wrapper
