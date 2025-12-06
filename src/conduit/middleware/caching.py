"""
This decorator wraps any function that sends a Request and receives a ConduitResult
(Response, ConduitError, SyncStream, AsyncStream).

If a Response is returned, it is persisted in a cache associated with the instance.
The cache key is derived from the Request by the cache implementation itself.

ConduitCache has get / set methods to retrieve and store Responses.
If a ConduitError or stream type is returned, it is not cached.

The cache object is exposed as self.cache_engine on Instrumentable objects.
"""

from collections.abc import Callable
import functools

from conduit.middleware.protocol import Instrumentable
from conduit.domain.result.response import Response
from conduit.domain.message.message import AssistantMessage
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import Request
from conduit.domain.result.result import ConduitResult


def cache_sync(
    func: Callable[[Request], ConduitResult],
) -> Callable[[Request], ConduitResult]:
    @functools.wraps(func)
    def sync_wrapper(*args: object, **kwargs: object) -> ConduitResult:
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

        # 2. Check Cache
        # if getattr(self, "cache_engine", None) is not None:
        #     cached = self.cache_engine.get(request)
        #     # Explicit miss semantics: cache.get must return None on miss
        #     if cached is not None:
        #         return cached

        # 3. Execute Original

        # Pre execute
        print("[Cache start]")

        result = func(*args, **kwargs)

        print("[Cache end]")
        # Post execute

        # 4. Store Result (only cache concrete Responses)
        # if (
        #     isinstance(result, Response)
        #     and getattr(self, "cache_engine", None) is not None
        # ):
        #     self.cache_engine.set(request, result)

        return result

    return sync_wrapper


def cache_async(
    func: Callable[[Request], ConduitResult],
) -> Callable[[Request], ConduitResult]:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> ConduitResult:
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

        # 2. Check Cache
        if getattr(self, "cache_engine", None) is not None:
            cached = self.cache_engine.get(request)
            if cached is not None:
                return cached

        # 3. Execute Original
        result = await func(*args, **kwargs)

        # 4. Store Result (only cache concrete Responses)
        if (
            isinstance(result, Response)
            and getattr(self, "cache_engine", None) is not None
        ):
            self.cache_engine.set(request, result)

        return result

    return async_wrapper


if __name__ == "__main__":
    params = GenerationParams(model="gpt3")
    req = Request(
        params=params,
        messages=[],
    )
    metadata = ResponseMetadata(
        model_slug="gpt3",
        duration=1.23,
        input_tokens=10,
        output_tokens=20,
        stop_reason=StopReason.STOP,
    )
    assistant_message = AssistantMessage(content="Hello, world!")
    response = Response(
        request=req,
        message=assistant_message,
        metadata=metadata,
    )

    class DummyCache:
        def __init__(self):
            self.store = {}
            self.get_calls = 0
            self.set_calls = 0

        def get(self, request: Request):
            self.get_calls += 1
            # Cache implementation is responsible for how Request is hashed.
            # For this dummy, we just use the object identity.
            return self.store.get(id(request), None)

        def set(self, request: Request, response: Response):
            self.set_calls += 1
            self.store[id(request)] = response

    class DummyInstrumentable:
        def __init__(self):
            self.cache_engine = DummyCache()

        @cache_sync
        def handle(self, request: Request) -> ConduitResult:
            # In your real code, this would be the expensive pipeline step.
            # For testing, we just return a new Response instance.
            return response

    dummy = DummyInstrumentable()

    # First call: should miss cache and call handle().
    res1 = dummy.handle(req)

    # Second call with the same Request: should hit cache.
    res2 = dummy.handle(req)

    # Basic assertions
    print("res1 is res2:", res1 is res2)
    print("cache get calls:", dummy.cache_engine.get_calls)
    print("cache set calls:", dummy.cache_engine.set_calls)
