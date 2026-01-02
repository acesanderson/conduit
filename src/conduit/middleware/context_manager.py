from __future__ import annotations
from contextlib import asynccontextmanager
import time
import logging
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.progress.utils import extract_query_preview

logger = logging.getLogger(__name__)


# _get_progress_handler helper stays the same...
def _get_progress_handler(request: GenerationRequest):
    if request.options.console is not None:
        from conduit.utils.progress.rich_handler import RichProgressHandler

        return RichProgressHandler(request.options.console)
    else:
        from conduit.utils.progress.plain_handler import PlainProgressHandler

        return PlainProgressHandler()


@asynccontextmanager
async def middleware_context_manager(request: GenerationRequest):
    """
    The Central Conductor for Request execution.
    Fully async.
    """
    if request.verbosity_override is not None:
        verbosity = request.verbosity_override
    else:
        verbosity = request.options.verbosity

    # --- 1. SETUP ---
    handler = _get_progress_handler(request)
    model_name = request.params.model
    preview = extract_query_preview(request.messages)

    ctx = {"cache_hit": False, "result": None}
    start_time = time.time()

    # --- 2. UI START ---
    if verbosity >= Verbosity.PROGRESS:
        handler.show_spinner(
            model_name=model_name,
            query_preview=preview,
            verbosity=verbosity,
        )

    # Debug print block stays same...

    # --- 3. CACHE READ (Async) ---
    if request.options.cache is not None and request.options.use_cache:
        # AWAIT HERE
        cached_result = await request.options.cache.get(request)
        if isinstance(cached_result, GenerationResponse):
            ctx["cache_hit"] = True
            ctx["result"] = cached_result
            logger.info("Cache hit.")

    # --- 4. EXECUTION (YIELD) ---
    yield ctx

    # --- 5. UI STOP (SUCCESS) ---
    duration = time.time() - start_time
    result = ctx.get("result")

    if verbosity >= Verbosity.PROGRESS:
        if ctx["cache_hit"]:
            handler.show_cached(
                model_name=model_name,
                query_preview=preview,
                duration=duration,
                verbosity=verbosity,
            )
        else:
            handler.show_complete(
                model_name=model_name,
                query_preview=preview,
                duration=duration,
                verbosity=verbosity,
                response_obj=result if isinstance(result, GenerationResponse) else None,
            )

    # --- 6. TEARDOWN (IO) ---
    if isinstance(result, GenerationResponse):
        # Cache Write (Async)
        if not ctx["cache_hit"] and request.options.cache is not None:
            logger.debug("Persisting result to cache.")
            # AWAIT HERE
            await request.options.cache.set(request, result)

        # Telemetry (Sync - Keep synchronous for Odometer)
        if not ctx["cache_hit"] and result.metadata.output_tokens > 0:
            from conduit.config import settings
            from conduit.storage.odometer.token_event import TokenEvent

            telemetry = settings.odometer_registry()
            event = TokenEvent(
                model=model_name,
                input_tokens=result.metadata.input_tokens,
                output_tokens=result.metadata.output_tokens,
            )
            telemetry.emit_token_event(event)
