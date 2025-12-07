from contextlib import contextmanager, asynccontextmanager
from conduit.domain.result.response import Response
from conduit.domain.request.request import Request
from conduit.middleware.protocol import Instrumentable
import logging

logger = logging.getLogger(__name__)

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
def middleware_context_manager(self: Instrumentable, request: Request):
    ctx = {}
    # 1. Pre-execute logic occurs in try block
    try:
        # Display logic
        logger.debug("Display logic begins...")
        # Cache logic
        logger.debug("Cache logic begins...")
        if self.cache is not None and request.use_cache is True:
            cached_result = self.cache.get(request)
            # If we have a cached Response, use it
            if isinstance(cached_result, Response):
                logger.info("Cache hit; using cached result.")
                ctx["cache_hit"] = True
                # Pass cached result back in context
                ctx["result"] = cached_result
            # No cached result
            else:
                logger.info("Cache miss; proceeding to execute request.")
                ctx["cache_hit"] = False
        else:
            logger.info(
                "Caching disabled or not configured; proceeding to execute request."
            )
            ctx["cache_hit"] = False
        yield ctx

    # 2. Post-execute logic occurs in finally block
    finally:
        # Cache store logic
        if self.cache is not None:
            result = ctx.get("result")
            if isinstance(result, Response):
                logger.info("Storing result in cache if applicable...")
                self.cache.set(request, result)
            else:
                logger.info("Result not cacheable; skipping cache store.")
        # Telemetry logic
        logger.debug("Telemetry logic begins...")
        # Display finalization logic
        logger.debug("Display finalization logic begins...")
        result = ctx.get("result")
