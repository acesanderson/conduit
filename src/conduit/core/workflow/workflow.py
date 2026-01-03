"""
OVERVIEW
--------
Workflow separates the DEFINITION of a workflow (code) from its OBSERVATION and
CONFIGURATION (runtime). It uses a "Harness" pattern to inject state via
ContextVars, keeping the domain logic pure.

CORE CONCEPTS
-------------
1. WORKFLOW (The Orchestrator)
   - A callable that defines the sequence of operations (A -> B -> C).
   - It is purely functional and stateless regarding infrastructure.

2. STEP (The Unit of Work)
   - A function decorated with `@step`.
   - It performs actual logic (LLM calls, data processing).
   - It automatically logs its inputs/outputs/latency to the Trace.
   - It pulls configuration via `get_param()` for runtime tuning.

3. STRATEGY (The Interchangeable Unit)
   - A Step that adheres to a strict interface (Protocol).
   - Used when a slot in a workflow can be filled by multiple implementations
     (e.g., 'Summarizer' can be 'FastSummarizer' or 'DeepSummarizer').

4. HARNESS (The Runtime)
   - Wraps the execution of a Workflow.
   - Manages the lifecycle of Trace and Config context variables.

TUNING & NAMESPACING
--------------------
Configuration is resolved using a specific precedence rule in `get_param()`:
   1. SCOPED:  "{step_name}.{param_key}" (e.g., "DraftEmail.model")
   2. GLOBAL:  "{param_key}"             (e.g., "model")
   3. DEFAULT: The value provided in code.

This allows you to set a global default (e.g., "gpt-3.5") while tuning specific
critical steps to use higher-performance models (e.g., "gpt-4") during evals.
"""

import functools
import inspect
import time
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

# --- INFRASTRUCTURE (THE "BUS") ---
# Holds the active configuration for the current run
_config_ctx = ContextVar("config", default={})
# Holds the trace log for the current run. If None, tracing is disabled.
_trace_ctx = ContextVar("trace", default=None)
# Holds a scratchpad for the CURRENTLY executing step -- for capturing things like token usage
_step_meta_ctx = ContextVar("step_meta", default=None)


# 2. NEW: The helper your strategies will use
def add_metadata(key: str, value: Any):
    """
    Log metadata to the current step (e.g., token usage, model version)
    without cluttering the return value.
    """
    meta = _step_meta_ctx.get()
    if meta is not None:
        meta[key] = value


def get_param(key: str, default: Any = None, scope: str | None = None) -> Any:
    """
    Retrieves a tunable parameter from the active context.

    Resolution Order:
    1. "{scope}.{key}" (Specific override)
    2. "{key}"         (Global override)
    3. default         (Hardcoded fallback)

    If `scope` is not provided, it attempts to infer the calling function's name.
    """
    cfg = _config_ctx.get()

    # Auto-detect scope if not provided
    if scope is None:
        # stack[1] is this function, stack[2] is the caller
        try:
            scope = inspect.stack()[1].function
        except (IndexError, AttributeError):
            scope = "unknown"

    # 1. Check Scoped
    scoped_key = f"{scope}.{key}"
    if scoped_key in cfg:
        return cfg[scoped_key]

    # 2. Check Global
    if key in cfg:
        return cfg[key]

    # 3. Default
    return default


def step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()

        # Create a fresh scratchpad for this step
        current_meta = {}
        token_meta = _step_meta_ctx.set(current_meta)

        try:
            # Run the function (which might call add_metadata)
            result = func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "error"
            raise e
        finally:
            duration = time.time() - start

            # Save to the global Trace Log
            trace_list = _trace_ctx.get()
            if trace_list is not None:
                trace_list.append(
                    {
                        "step": func.__name__,
                        "inputs": {"args": args, "kwargs": kwargs},  # Simplify for prod
                        "output": result,
                        "duration": round(duration, 4),
                        "status": status,
                        # HERE IS YOUR METADATA (Tokens, etc.)
                        "metadata": current_meta,
                    }
                )

            # Clean up the scratchpad (ContextVars handles nesting automatically!)
            _step_meta_ctx.reset(token_meta)

        return result

    return wrapper


# --- PROTOCOL DEFINITIONS ---
@runtime_checkable
class Step(Protocol):
    """
    A unit of work. Must be decorated with @step.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class Strategy(Protocol):
    """
    A Step with a standardized signature for interchangeability.
    """

    def __call__(self, input_data: Any) -> Any: ...


@runtime_checkable
class Workflow(Protocol):
    """
    The orchestrator script.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class ConduitHarness:
    """
    The runtime container that manages Observability and Configuration contexts.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.trace_log: list = []

    def run(self, workflow: Workflow, *args, **kwargs) -> Any:
        """
        Executes a workflow within the managed context.
        """
        # Mount Context
        token_conf = _config_ctx.set(self.config)
        token_trace = _trace_ctx.set(self.trace_log)

        try:
            return workflow(*args, **kwargs)
        finally:
            # Unmount Context
            _config_ctx.reset(token_conf)
            _trace_ctx.reset(token_trace)

    @property
    def trace(self) -> list:
        return self.trace_log


if __name__ == "__main__":
    # --- EXAMPLE USAGE ---
    @step
    def DraftEmail(topic: str) -> str:
        model = get_param("model", default="gpt-3.5")
        # result = Model(model).query(f"Draft an email about {topic}")
        add_metadata("model_used", model)
        add_metadata("input_tokens", 50)
        add_metadata("output_tokens", 150)
        return f"Drafted email about '{topic}' using model {model}."

    @step
    def SummarizeEmail(draft: str) -> str:
        model = get_param("model", default="gpt-3.5")
        length = get_param("length", default="short")
        # result = Model(model).query(f"Summarize this email in a {length} format: {draft}")
        add_metadata("model_used", model)
        add_metadata("input_tokens", 150)
        add_metadata("output_tokens", 50)
        return f"Summarized ({length}): {draft}"

    def EmailWorkflow(topic: str) -> str:
        draft = DraftEmail(topic)
        summary = SummarizeEmail(draft)
        return summary

    harness = ConduitHarness(
        config={
            "DraftEmail.model": "gpt-4",
            "SummarizeEmail.model": "gp3",
            "length": "detailed",
        }
    )

    result = harness.run(EmailWorkflow, topic="AI in Healthcare")
