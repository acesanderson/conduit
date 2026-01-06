import functools
import inspect
import time
from typing import Any

from conduit.core.workflow.context import context


def add_metadata(key: str, value: Any):
    """
    Log metadata to the current step (e.g., token usage, model version)
    without cluttering the return value.
    """
    meta = context.step_meta.get()
    if meta is not None:
        meta[key] = value


def resolve_param(
    key: str, default: Any, kwargs: dict, scope: str | None = None
) -> Any:
    """
    Resolves a parameter value in strict precedence:
    1. Runtime Override (kwargs passed explicitly to the function)
    2. Harness Configuration (ContextVars via get_param)
    3. Hardcoded Default (provided as arg)

    Side Effects:
    - Logs the resolution source to the active step's metadata (traceability).
    - Registers the parameter in the discovery log (visibility).
    """
    resolution_source = "default"
    value = default
    active_key = None  # The key actually read from config, if any

    # Auto-detect scope if not provided (fallback)
    if scope is None:
        try:
            scope = inspect.stack()[1].function
        except (IndexError, AttributeError):
            scope = "unknown"

    # --- 1. RESOLUTION LOGIC ---

    # 1. Runtime Override (Explicit wins)
    if key in kwargs:
        value = kwargs[key]
        resolution_source = "runtime_kwarg"

    # 2. Harness Config
    else:
        cfg = context.config.get()
        # Check Scoped
        if scope:
            scoped_key = f"{scope}.{key}"
            if scoped_key in cfg:
                value = cfg[scoped_key]
                resolution_source = f"config_scoped ({scoped_key})"
                active_key = scoped_key

        # Check Global (only if not found in scope)
        if resolution_source == "default" and key in cfg:
            value = cfg[key]
            resolution_source = f"config_global ({key})"
            active_key = key

    # --- 2. TELEMETRY (Trace) ---
    # Log the decision to the current step's metadata
    meta = context.step_meta.get()
    if meta is not None:
        if "config_resolutions" not in meta:
            meta["config_resolutions"] = {}

        meta["config_resolutions"][key] = {
            "value": str(value),  # stringify to ensure JSON serializable
            "source": resolution_source,
        }

    # --- 3. DISCOVERY (Schema) ---
    # Log that this key was requested, along with its default value.
    discovery_log = context.discovery.get()
    if discovery_log is not None:
        # Construct a canonical key for the log (preferring scoped)
        canonical_key = f"{scope}.{key}" if scope else key
        # We store the default value so the user knows what the baseline is
        discovery_log[canonical_key] = default

    # --- 4. DRIFT DETECTION ---
    if "config" in resolution_source and active_key:
        accessed_keys = context.access.get()
        if accessed_keys is not None:
            accessed_keys.add(active_key)

    return value


def get_param(key: str, default: Any = None, scope: str | None = None) -> Any:
    """
    Legacy wrapper for resolve_param.
    Used by steps that haven't migrated to explicit kwargs injection yet.
    """
    return resolve_param(key, default, {}, scope)


def step(func):
    if not inspect.iscoroutinefunction(func):
        raise TypeError(
            f"@step requires async functions. {func.__name__} must be defined with `async def`."
        )

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()

        # Initialize scratchpad for this step
        current_meta = {}
        token_meta = context.step_meta.set(current_meta)

        try:
            result = await func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "error"
            raise
        finally:
            duration = time.time() - start

            # Append to the global trace
            trace_list = context.trace.get()
            if trace_list is not None:
                trace_list.append(
                    {
                        "step": f"{func.__qualname__.replace('.__call__', '')}",
                        "inputs": {"args": args, "kwargs": kwargs},
                        "output": result,
                        "duration": round(duration, 4),
                        "status": status,
                        "metadata": current_meta,
                    }
                )

            # Reset scratchpad
            context.step_meta.reset(token_meta)

        return result

    return wrapper
