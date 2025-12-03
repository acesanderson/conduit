"""
This decorator wraps any function that sends a Request and receives a Response or a ConduitError.
If a Response is returned, it is persisted in a cache with a key based on the hash of the Request.
ConduitCache has get / set methods to retrieve and store Responses.
If a ConduitError is returned, it is not cached.
The ConduitCache object is self.cache, which must be present on Instrumentable objects.
"""

from collections.abc import Callable
from conduit.middleware.protocol import Instrumentable
from conduit.domain.result.response import Response
from conduit.domain.request.request import Request
from conduit.domain.result.result import ConduitResult
import functools
import inspect


def cache(func: Callable) -> Callable:
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

            # 2. Check Cache
            if self.cache_engine:
                # Assuming cache engine has specific async implementation
                cached = await self.cache_engine.get_async(request)
                if cached:
                    return cached

            # 3. Execute Original
            result = await func(*args, **kwargs)

            # 4. Store Result (only Responses)
            if isinstance(result, Response) and self.cache_engine:
                await self.cache_engine.set_async(request, result)

            return result

        return async_wrapper

    # Sync wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> ConduitResult:
            # 1. Validate & Extract
            self, request = _extract_args(args, kwargs)

            # 2. Check Cache
            if self.cache_engine:
                cached = self.cache_engine.get(request)
                if cached:
                    return cached

            # 3. Execute Original
            result = func(*args, **kwargs)

            # 4. Store Result
            if isinstance(result, Response) and self.cache_engine:
                self.cache_engine.set(request, result)

            return result

        return sync_wrapper
