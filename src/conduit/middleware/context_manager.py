from contextlib import contextmanager
from typing import TYPE_CHECKING
import time
import logging
from conduit.domain.result.response import Response
from conduit.domain.request.request import Request
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.progress.utils import extract_query_preview

if TYPE_CHECKING:
    from conduit.middleware.protocol import Instrumentable

logger = logging.getLogger(__name__)


def _get_progress_handler(self: "Instrumentable"):
    """Factory to get the correct handler based on instance configuration."""
    if getattr(self, "console", None):
        from conduit.utils.progress.rich_handler import RichProgressHandler

        return RichProgressHandler(self.console)
    else:
        from conduit.utils.progress.plain_handler import PlainProgressHandler

        return PlainProgressHandler()


@contextmanager
def middleware_context_manager(self: "Instrumentable", request: Request):
    """
    The Central Conductor for Request execution.

    Lifecycle:
    1. Setup: Prepare UI handler, timestamps, and query preview.
    2. UI Start: Trigger the spinner.
    3. Cache Read: Check for existing result.
    4. Execution (Yield): Run the Model's logic (if no cache hit).
    5. UI Stop: Trigger Success/Cached/Error display.
    6. Teardown: Write to cache and emit telemetry.
    """

    # --- 1. SETUP ---
    handler = _get_progress_handler(self)
    model_name = request.params.model
    # Extract prompt string from the request messages for the UI
    preview = extract_query_preview(request.messages)

    # Initialize context state to track flow across the yield boundary
    ctx = {"cache_hit": False, "result": None}

    start_time = time.time()

    # We use a nested try/finally block to ensure the UI spinner stops
    # and errors are displayed correctly even if the API call explodes.
    try:
        # --- 2. UI START ---
        # Only show spinner if verbosity allows
        if request.verbosity >= Verbosity.PROGRESS:
            handler.show_spinner(
                model_name=model_name,
                query_preview=preview,
                verbosity=request.verbosity,
            )

        # --- 3. CACHE READ ---
        # We check this *before* yielding to avoid unnecessary API calls
        if self.cache is not None and request.use_cache:
            cached_result = self.cache.get(request)
            if isinstance(cached_result, Response):
                ctx["cache_hit"] = True
                ctx["result"] = cached_result
                logger.info("Cache hit.")

        # --- 4. EXECUTION (YIELD) ---
        # We yield the context. The @middleware_sync wrapper will see `ctx`
        # and skip the actual function call if `ctx["result"]` is populated.
        yield ctx

    except Exception as e:
        # --- ERROR HANDLING ---
        duration = time.time() - start_time
        # Update UI to show Red X
        if request.verbosity >= Verbosity.PROGRESS:
            handler.show_failed(
                model_name=model_name,
                query_preview=preview,
                error=str(e),
                verbosity=request.verbosity,
                error_obj=e
                if hasattr(e, "info")
                else None,  # Handle ConduitError vs generic
            )
        raise e

    # --- 5. UI STOP (SUCCESS) ---
    duration = time.time() - start_time
    result = ctx.get("result")

    # Determine how to display the result (Lightning Bolt vs Green Check)
    if request.verbosity >= Verbosity.PROGRESS:
        if ctx["cache_hit"]:
            handler.show_cached(
                model_name=model_name,
                query_preview=preview,
                duration=duration,
                verbosity=request.verbosity,
            )
        else:
            handler.show_complete(
                model_name=model_name,
                query_preview=preview,
                duration=duration,
                verbosity=request.verbosity,
                response_obj=result if isinstance(result, Response) else None,
            )

    # --- 6. TEARDOWN (IO) ---
    # Only perform expensive IO if we actually generated a new Response
    if isinstance(result, Response):
        # Cache Write (if this was a new generation)
        if not ctx["cache_hit"] and self.cache is not None:
            logger.debug("Persisting result to cache.")
            self.cache.set(request, result)

        # Telemetry / Odometer (always record usage, even if cached?)
        # Usually we only record 'spend' on non-cached hits, but we might want
        # to track 'savings' on cached hits. For now, track actual API usage.
        if not ctx["cache_hit"] and result.metadata.output_tokens > 0:
            if getattr(self, "odometer_registry", None):
                from conduit.storage.odometer.token_event import TokenEvent

                event = TokenEvent(
                    model=model_name,
                    input_tokens=result.metadata.input_tokens,
                    output_tokens=result.metadata.output_tokens,
                )
                self.odometer_registry.emit_token_event(event)
