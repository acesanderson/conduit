from collections.abc import Callable, Awaitable
from conduit.middleware.protocol import Instrumentable
from conduit.domain.result.response import Response
from conduit.domain.request.request import Request
from conduit.domain.result.result import ConduitResult
import functools
import inspect


def progress(
    func: Callable[[Request], ConduitResult],
) -> Callable[..., Awaitable[ConduitResult]] | Callable[..., ConduitResult]:
    def _extract_args(args: tuple, kwargs: dict) -> tuple[Instrumentable, Request]:
        # Extract self
        if not args:
            raise TypeError("Decorator expects a bound method (missing 'self').")

        instance = args[0]
        if not isinstance(instance, Instrumentable):
            raise TypeError(
                f"The cache decorator can only be applied to methods of Instrumentable classes. Got {type(instance).__name__}"
            )

        # Extract request; It might be the second positional arg, OR a keyword arg
        if len(args) > 1:
            req = args[1]
        elif "request" in kwargs:
            req = kwargs["request"]
        else:
            raise TypeError(
                f"Could not find 'request' argument in {func.__name__} call."
            )

        if not isinstance(req, Request):
            raise TypeError(
                f"Argument 'request' must be of type Request. Got {type(req)}."
            )

        return instance, req

    # ASYNC Wrapper
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> ConduitResult:
            # 1. Validate & Extract
            self, request = _extract_args(args, kwargs)

            ...

            return result

        return async_wrapper

    # Sync wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> ConduitResult:
            # 1. Validate & Extract
            self, request = _extract_args(args, kwargs)

            ...

            return result

        return sync_wrapper
