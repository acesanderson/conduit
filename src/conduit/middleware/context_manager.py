from contextlib import contextmanager, asynccontextmanager
from conduit.domain.result.result import ConduitResult

"""
What happens pre-execute:
- display progress status / set up spinner (per request.verbosity level)
- check cache for existing result (submit Request object to self.cache.get)

What happens post-execute:
- display progress
    - if result = Response, display response data
    - if result = ConduitError, display error message
    - if result = SyncStream or AsyncStream, 
- store result in cache (if isinstance(result, Response))
- send telemetry data (if isinstance(result, Response))
"""


@contextmanager
def middleware_context_manager(self, request) -> dict[str, object]:
    ctx = {}
    try:
        print("[DISPLAY] Progess blah blah...")
        print("[CHECK CACHE] Checking cache for existing result...")
        # if cache_hit, ctx["result"] = cached_result
        yield ctx
    finally:
        print("[STORE CACHE] Storing result in cache...")
        print("[TELEMETRY] Sending telemetry data...")
        print("[DISPLAY] Finalizing progress display...")
        result = ctx.get("result")
        pass
