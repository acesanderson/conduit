import asyncio
import warnings


def _warn_if_loop_exists():
    try:
        asyncio.get_running_loop()
        warnings.warn(
            "Blocking call detected inside an event loop. Use the Async client or ConduitBatch to avoid blocking the main thread.",
            RuntimeWarning,
        )
    except RuntimeError:
        pass
