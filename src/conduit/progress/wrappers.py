"""
Enhanced progress wrappers with verbosity support.
Maintains backwards compatibility while adding new verbosity parameter.
"""

from conduit.logs.logging_config import get_logger
from conduit.progress.verbosity import Verbosity
from functools import wraps
import time, sys, inspect

logger = get_logger(__name__)


def extract_query_preview(input_data, max_length=30):
    """Extract a preview of the query for display purposes"""
    if isinstance(input_data, str):
        # Strip whitespace and replace newlines with spaces
        preview = input_data.strip().replace("\n", " ").replace("\r", " ")
        return preview[:max_length] + "..." if len(preview) > max_length else preview
    elif isinstance(input_data, list):
        # Find last message with role="user"
        for message in reversed(input_data):
            if hasattr(message, "role") and message.role == "user":
                content = message.content
                # Handle Pydantic objects
                if hasattr(content, "model_dump_json"):
                    content = content.model_dump_json()
                else:
                    content = str(content)
                # Strip whitespace and replace newlines with spaces
                content = content.strip().replace("\n", " ").replace("\r", " ")
                return (
                    content[:max_length] + "..."
                    if len(content) > max_length
                    else content
                )
        return "No user message found"
    else:
        preview = str(input_data).strip().replace("\n", " ").replace("\r", " ")
        return preview[:max_length] + "..."


def sync_wrapper(
    model_instance,
    func,
    handler,
    query_preview,
    index,
    total,
    verbosity,
    *args,
    **kwargs,
):
    """Synchronous wrapper for progress display with verbosity support"""
    model_name = model_instance.model
    display_preview = (
        f"[{index}/{total}] {query_preview}" if index is not None else query_preview
    )

    # Show starting state (only for PROGRESS and above)
    if verbosity >= Verbosity.PROGRESS:
        if hasattr(handler, "show_spinner"):
            handler.show_spinner(model_name, display_preview, verbosity=verbosity)
        else:
            handler.emit_started(model_name, display_preview, verbosity=verbosity)

    start_time = time.time()
    try:
        result = func(model_instance, *args, **kwargs)
        duration = time.time() - start_time

        # Only show completion for PROGRESS and above
        if verbosity >= Verbosity.PROGRESS:
            # Check if result is an error
            from conduit.result.error import ConduitError

            if isinstance(result, ConduitError):
                if hasattr(handler, "show_failed"):
                    handler.show_failed(
                        model_name,
                        display_preview,
                        str(result.info.message),
                        verbosity=verbosity,
                        error_obj=result,
                    )
                else:
                    handler.emit_failed(
                        model_name,
                        display_preview,
                        str(result.info.message),
                        verbosity=verbosity,
                    )
            else:
                # Check for cache hit (very fast response)
                is_cache_hit = duration < 0.05  # Less than 50ms indicates cache hit

                if is_cache_hit:
                    if hasattr(handler, "show_cached"):
                        handler.show_cached(
                            model_name, display_preview, duration, verbosity=verbosity
                        )
                    else:
                        handler.emit_cached(
                            model_name, display_preview, duration, verbosity=verbosity
                        )
                else:
                    # Normal success case
                    if hasattr(handler, "show_complete"):
                        handler.show_complete(
                            model_name,
                            display_preview,
                            duration,
                            verbosity=verbosity,
                            response_obj=result,
                        )
                    else:
                        handler.emit_complete(
                            model_name, display_preview, duration, verbosity=verbosity
                        )

        return result
    except KeyboardInterrupt:
        if verbosity >= Verbosity.PROGRESS:
            if hasattr(handler, "show_canceled"):
                handler.show_canceled(model_name, display_preview, verbosity=verbosity)
            else:
                handler.emit_canceled(model_name, display_preview, verbosity=verbosity)

        # Exit gracefully without stack trace
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        if verbosity >= Verbosity.PROGRESS:
            if hasattr(handler, "show_failed"):
                handler.show_failed(
                    model_name, display_preview, str(e), verbosity=verbosity
                )
            else:
                handler.emit_failed(
                    model_name, display_preview, str(e), verbosity=verbosity
                )
        raise


async def async_wrapper(
    model_instance, func, handler, query_preview, verbosity, *args, **kwargs
):
    """Asynchronous wrapper for progress display with verbosity support"""
    model_name = model_instance.model

    # Show starting state (only for PROGRESS and above)
    if verbosity >= Verbosity.PROGRESS:
        if hasattr(handler, "show_spinner"):
            handler.show_spinner(model_name, query_preview, verbosity=verbosity)
        else:
            handler.emit_started(model_name, query_preview, verbosity=verbosity)

    start_time = time.time()
    try:
        result = await func(model_instance, *args, **kwargs)
        duration = time.time() - start_time

        # Only show completion for PROGRESS and above
        if verbosity >= Verbosity.PROGRESS:
            # Check if result is an error
            from conduit.result.error import ConduitError

            if isinstance(result, ConduitError):
                if hasattr(handler, "show_failed"):
                    handler.show_failed(
                        model_name,
                        query_preview,
                        str(result.info.message),
                        verbosity=verbosity,
                        error_obj=result,
                    )
                else:
                    handler.emit_failed(
                        model_name,
                        query_preview,
                        str(result.info.message),
                        verbosity=verbosity,
                    )
            else:
                # Check for cache hit (very fast response)
                is_cache_hit = duration < 0.05  # Less than 50ms indicates cache hit

                if is_cache_hit:
                    if hasattr(handler, "show_cached"):
                        handler.show_cached(
                            model_name, query_preview, duration, verbosity=verbosity
                        )
                    else:
                        handler.emit_cached(
                            model_name, query_preview, duration, verbosity=verbosity
                        )
                else:
                    # Normal success case
                    if hasattr(handler, "show_complete"):
                        handler.show_complete(
                            model_name,
                            query_preview,
                            duration,
                            verbosity=verbosity,
                            response_obj=result,
                        )
                    else:
                        handler.emit_complete(
                            model_name, query_preview, duration, verbosity=verbosity
                        )

        return result

    except KeyboardInterrupt:
        if verbosity >= Verbosity.PROGRESS:
            if hasattr(handler, "show_canceled"):
                handler.show_canceled(model_name, query_preview, verbosity=verbosity)
            else:
                handler.emit_canceled(model_name, query_preview, verbosity=verbosity)

        # Exit gracefully without stack trace
        sys.exit(130)  # Standard exit code for Ctrl+C

    except Exception as e:
        if verbosity >= Verbosity.PROGRESS:
            if hasattr(handler, "show_failed"):
                handler.show_failed(
                    model_name, query_preview, str(e), verbosity=verbosity
                )
            else:
                handler.emit_failed(
                    model_name, query_preview, str(e), verbosity=verbosity
                )
        raise


def progress_display(func):
    """
    Decorator that adds progress display to Model.query() methods with verbosity support.
    Automatically detects sync vs async and uses appropriate wrapper.
    """

    @wraps(func)
    def sync_decorator(self, *args, **kwargs):
        # Extract and convert verbosity parameter
        verbose_input = kwargs.pop("verbose", True)
        verbosity = Verbosity.from_input(verbose_input)

        index = kwargs.pop("index", None)
        total = kwargs.pop("total", None)

        # Validate index/total parameters
        if (index is None) != (total is None):
            raise ValueError(
                "Must provide both 'index' and 'total' parameters or neither"
            )

        # For SILENT, bypass all progress display
        if verbosity == Verbosity.SILENT:
            return func(self, *args, **kwargs)

        # Extract query preview from first argument
        query_preview = extract_query_preview(args[0] if args else "")

        if self.console:
            # Lazy import Rich components only when needed
            from conduit.progress.handlers import RichProgressHandler

            handler = RichProgressHandler(self.console)
        else:
            # Built-in PlainProgressHandler - no imports
            from conduit.progress.handlers import PlainProgressHandler

            handler = PlainProgressHandler()

        return sync_wrapper(
            self, func, handler, query_preview, index, total, verbosity, *args, **kwargs
        )

    @wraps(func)
    async def async_decorator(self, *args, **kwargs):
        # Extract and convert verbosity parameter
        verbose_input = kwargs.pop("verbose", True)
        verbosity = Verbosity.from_input(verbose_input)

        # For SILENT, bypass all progress display
        if verbosity == Verbosity.SILENT:
            return await func(self, *args, **kwargs)

        # Extract query preview from first argument
        query_preview = extract_query_preview(args[0] if args else "")

        if self.console:
            # Lazy import Rich components only when needed
            from conduit.progress.handlers import RichProgressHandler

            handler = RichProgressHandler(self.console)
        else:
            # Built-in PlainProgressHandler - no imports
            from conduit.progress.handlers import PlainProgressHandler

            handler = PlainProgressHandler()

        return await async_wrapper(
            self, func, handler, query_preview, verbosity, *args, **kwargs
        )

    # Return the appropriate decorator based on function type
    if inspect.iscoroutinefunction(func):
        return async_decorator
    else:
        return sync_decorator


async def concurrent_wrapper(operation, tracker):
    """Wrap individual async operations for concurrent tracking"""
    try:
        tracker.operation_started()
        result = await operation
        tracker.operation_completed()
        return result
    except Exception as e:
        tracker.operation_failed()
        raise  # Re-raise the exception


def create_concurrent_progress_tracker(console, total: int):
    """Factory function to create appropriate concurrent tracker"""
    if console:
        from conduit.progress.handlers import RichProgressHandler

        handler = RichProgressHandler(console)
    else:
        from conduit.progress.handlers import PlainProgressHandler

        handler = PlainProgressHandler()

    from conduit.progress.tracker import ConcurrentTracker

    return ConcurrentTracker(handler, total)
