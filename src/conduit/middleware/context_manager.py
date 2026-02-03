from __future__ import annotations
from contextlib import asynccontextmanager
import time
import logging
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.progress.utils import extract_query_preview
from conduit.core.workflow.context import context

logger = logging.getLogger(__name__)


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

    # --- 3. CACHE READ (Async) ---
    if request.options.cache is not None and request.options.use_cache:
        cached_result = await request.options.cache.get(request)
        if isinstance(cached_result, GenerationResponse):
            ctx["cache_hit"] = True
            ctx["result"] = cached_result
            # Reconstruct parsed field for structured responses
            if request.params.response_model and cached_result.message.content:
                cached_result.message.parsed = request.params.response_model.model_validate_json(
                    cached_result.message.content
                )
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

    # --- 6. TEARDOWN (IO & TELEMETRY) ---
    if isinstance(result, GenerationResponse):
        # A. Cache Write (Async)
        if not ctx["cache_hit"] and request.options.cache is not None:
            logger.debug("Persisting result to cache.")
            await request.options.cache.set(request, result)

        # B. Odometer Telemetry (Async Flush)
        if not ctx["cache_hit"] and result.metadata.output_tokens > 0:
            from conduit.config import settings
            from conduit.storage.odometer.token_event import TokenEvent

            telemetry = settings.odometer_registry()

            # Record to memory (Sync)
            event = TokenEvent(
                model=model_name,
                input_tokens=result.metadata.input_tokens,
                output_tokens=result.metadata.output_tokens,
            )
            telemetry.emit_token_event(event)

            # Trigger Async Flush
            await telemetry.flush()
            await telemetry.recover()

        # C. Workflow Trace Injection (The "Telemetric Middleware")
        # If we are currently running inside a @step, auto-log the token usage.
        if context.is_active:
            meta = context.step_meta.get()
            if meta is not None:
                # Initialize counters if this step makes multiple model calls
                meta.setdefault("input_tokens", 0)
                meta.setdefault("output_tokens", 0)
                meta.setdefault("model_calls", 0)

                # Accumulate tokens
                meta["input_tokens"] += result.metadata.input_tokens
                meta["output_tokens"] += result.metadata.output_tokens
                meta["model_calls"] += 1

                # Track unique models used in this step
                current_models = meta.get("models_used", [])
                if model_name not in current_models:
                    current_models.append(model_name)
                meta["models_used"] = current_models
