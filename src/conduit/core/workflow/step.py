import functools
import inspect
import time
from typing import Any, TypeVar, ParamSpec, cast
from collections.abc import Callable, Awaitable

from conduit.core.workflow.context import context

# Typing vars for the decorator
P = ParamSpec("P")
R = TypeVar("R")


def add_metadata(key: str, value: Any):
    """
    Log metadata to the current step (e.g., token usage, model version)
    without cluttering the return value.
    """
    meta = context.step_meta.get()
    if meta is not None:
        meta[key] = value


def resolve_param(
    key: str,
    default: Any,
    overrides: dict | None = None,
    scope: str | None = None,
) -> Any:
    """
    Resolves a parameter value in strict precedence:
    1. Runtime Override (Explicit argument passed to the function, if not None)
    2. Harness Configuration (ContextVars via get_param)
    3. Hardcoded Default (provided as arg)

    Args:
        key: The parameter name to look up (e.g., "model").
        default: The fallback value if found nowhere else.
        overrides: Optional manual dictionary to check first. If None, uses the
                   automatically captured arguments from the current @step.
        scope: Optional namespace for config lookup (defaults to caller name).
    """
    resolution_source = "default"
    value = default
    active_key = None  # The key actually read from config, if any

    # Auto-detect scope if not provided
    if scope is None:
        try:
            # Stack: [0] this function, [1] caller
            scope = inspect.stack()[1].function
        except (IndexError, AttributeError):
            scope = "unknown"

    # --- 1. RUNTIME OVERRIDE LOOKUP ---
    # We check 'overrides' if provided, otherwise check the context
    # populated by the @step decorator's auto-binding logic.
    runtime_args = overrides
    if runtime_args is None:
        runtime_args = context.args.get() or {}

    if key in runtime_args:
        # Crucial: We only accept the runtime value if it is NOT None.
        # This supports the pattern `def func(model=None): resolve_param("model", ...)`
        # allowing the None default to signal "fall through to config".
        if runtime_args[key] is not None:
            value = runtime_args[key]
            resolution_source = "runtime_kwarg"

    # --- 2. HARNESS CONFIG LOOKUP ---
    # Only check config if we haven't found a runtime override (or it was None)
    if resolution_source == "default":
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

    # --- 3. TELEMETRY (Trace) ---
    # Log the decision to the current step's metadata
    meta = context.step_meta.get()
    if meta is not None:
        if "config_resolutions" not in meta:
            meta["config_resolutions"] = {}

        meta["config_resolutions"][key] = {
            "value": str(value),  # stringify to ensure JSON serializable
            "source": resolution_source,
        }

    # --- 4. DISCOVERY (Schema) ---
    # Log that this key was requested, along with its default value.
    discovery_log = context.discovery.get()
    if discovery_log is not None:
        canonical_key = f"{scope}.{key}" if scope else key
        discovery_log[canonical_key] = default

    # --- 5. DRIFT DETECTION ---
    if "config" in resolution_source and active_key:
        accessed_keys = context.access.get()
        if accessed_keys is not None:
            accessed_keys.add(active_key)

    return value


def get_param(key: str, default: Any = None, scope: str | None = None) -> Any:
    """
    Legacy wrapper for resolve_param.
    """
    return resolve_param(key, default, None, scope)


def step(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    # 1. Capture the signature once at definition time
    if not inspect.iscoroutinefunction(func):
        raise TypeError(
            f"@step requires async functions. {func.__name__} must be defined with `async def`."
        )
    sig = inspect.signature(func)

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.time()

        # 2. AUTO-BINDING MAGIC
        # Bind args/kwargs to the function signature to normalize input
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Flatten into a single dict: {"text": "...", "model": None, ...}
        unified_args = dict(bound.arguments)

        # Handle **kwargs in signature: flatten nested dict if present
        # Find which param name holds the var_keyword (usually "kwargs")
        var_keyword_name = next(
            (
                p.name
                for p in sig.parameters.values()
                if p.kind == inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )
        if var_keyword_name and var_keyword_name in unified_args:
            extra = unified_args.pop(var_keyword_name)
            if extra:
                unified_args.update(extra)

        # 3. Mount Context
        token_args = context.args.set(unified_args)
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

            trace_list = context.trace.get()
            if trace_list is not None:
                trace_list.append(
                    {
                        "step": f"{func.__qualname__.replace('.__call__', '')}",
                        "inputs": unified_args,  # Log normalized inputs!
                        "output": result,
                        "duration": round(duration, 4),
                        "status": status,
                        "metadata": current_meta,
                    }
                )

            # Cleanup
            context.args.reset(token_args)
            context.step_meta.reset(token_meta)

        return result

    return cast(Callable[P, Awaitable[R]], wrapper)
