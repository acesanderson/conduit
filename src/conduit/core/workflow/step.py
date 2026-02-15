"""
Workflow step decorator and parameter resolution system for Conduit's async task orchestration.

This module provides core infrastructure for the `@step` decorator—enabling automatic input binding,
context variable management, and telemetry collection within async workflows. It also injects
introspection capabilities (.diagram, .schema) as properties on decorated functions.
"""

import ast
import functools
import inspect
import textwrap
import time
from collections import deque
from collections.abc import Callable, Awaitable
from typing import Any, TypeVar, ParamSpec, cast

from conduit.core.workflow.context import context

# Typing vars
P = ParamSpec("P")
R = TypeVar("R")


# --- 1. CORE UTILITIES ---


def resolve_param(
    key: str,
    default: Any,
    overrides: dict | None = None,
    scope: str | None = None,
) -> Any:
    """
    Resolves a parameter value in strict precedence:
    1. Runtime Override (Explicit argument passed to the function)
    2. Harness Configuration (ContextVars via get_param)
    3. Hardcoded Default (provided as arg)
    """
    resolution_source = "default"
    value = default
    active_key = None

    if scope is None:
        try:
            scope = inspect.stack()[1].function
        except (IndexError, AttributeError):
            scope = "unknown"

    # 1. RUNTIME OVERRIDE
    runtime_args = overrides
    if runtime_args is None:
        runtime_args = context.args.get() or {}

    if key in runtime_args and runtime_args[key] is not None:
        value = runtime_args[key]
        resolution_source = "runtime_kwarg"

    # 2. CONFIG LOOKUP
    if resolution_source == "default":
        cfg = context.config.get()
        if scope:
            scoped_key = f"{scope}.{key}"
            if scoped_key in cfg:
                value = cfg[scoped_key]
                resolution_source = f"config_scoped ({scoped_key})"
                active_key = scoped_key

        if resolution_source == "default" and key in cfg:
            value = cfg[key]
            resolution_source = f"config_global ({key})"
            active_key = key

    # 3. TELEMETRY
    meta = context.step_meta.get()
    if meta is not None:
        if "config_resolutions" not in meta:
            meta["config_resolutions"] = {}
        meta["config_resolutions"][key] = {
            "value": str(value),
            "source": resolution_source,
        }

    # 4. DISCOVERY
    discovery_log = context.discovery.get()
    if discovery_log is not None:
        canonical_key = f"{scope}.{key}" if scope else key
        discovery_log[canonical_key] = default

    # 5. DRIFT DETECTION
    if "config" in resolution_source and active_key:
        accessed = context.access.get()
        if accessed is not None:
            accessed.add(active_key)

    return value


def get_param(key: str, default: Any = None, scope: str | None = None) -> Any:
    return resolve_param(key, default, None, scope)


def add_metadata(key: str, value: Any) -> None:
    """
    Add metadata to the current step's execution trace.
    """
    meta = context.step_meta.get()
    if meta is not None:
        meta[key] = value


# --- 2. STATIC ANALYSIS LOGIC ---


def _static_scan_workflow(root_func) -> dict:
    schema = {}
    visited = set()
    root = getattr(root_func, "__wrapped__", root_func)
    queue = deque([root])
    visited.add(root)

    while queue:
        current_func = queue.popleft()
        try:
            # 1. Get raw source
            raw_src = inspect.getsource(current_func)
            # 2. Aggressively dedent so the function starts at column 0
            src = textwrap.dedent(raw_src)
            # 3. Parse the now-valid syntax tree
            tree = ast.parse(src)

            func_scope = getattr(current_func, "__globals__", {}).copy()
            # If it's a method, we need the class's namespace
            if inspect.ismethod(current_func) or "." in current_func.__qualname__:
                # Attempt to find the class owning this method
                cls_name = current_func.__qualname__.split(".")[0]
                # This is a bit of a hack, but standard for static analysis:
                # we look for the class in the module where the function was defined.
                module = inspect.getmodule(current_func)
                cls = getattr(module, cls_name, None)
                if cls:
                    func_scope.update(cls.__dict__)
        except (OSError, TypeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 1. Detect direct resolve_param calls
                call_id = ""
                if isinstance(node.func, ast.Name):
                    call_id = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_id = node.func.attr

                if call_id in ("resolve_param", "get_param"):
                    if len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                        key = node.args[0].value
                        # Default parsing logic...
                        schema[f"{current_func.__name__}.{key}"] = "dynamic"

                # 2. Recursive Search: Follow the call tree
                target_obj = func_scope.get(call_id)
                if target_obj:
                    # Follow both standard decorators and our StepWrapper
                    underlying = getattr(target_obj, "__wrapped__", target_obj)
                    if callable(underlying) and underlying not in visited:
                        visited.add(underlying)
                        queue.append(underlying)
    return schema


def _generate_hierarchy_graph(root_func) -> str:
    visited_funcs = set()
    root = getattr(root_func, "__wrapped__", root_func)
    queue = deque([root])
    edges = []

    while queue:
        current_func = queue.popleft()
        if current_func in visited_funcs:
            continue
        visited_funcs.add(current_func)

        parent_name = current_func.__name__
        try:
            src = inspect.getsource(current_func)
            src = inspect.cleandoc(src)
            tree = ast.parse(src)
            func_globals = getattr(current_func, "__globals__", {})
        except (OSError, TypeError):
            continue

        found_deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target_name = None
                if isinstance(node.func, ast.Name):
                    target_name = node.func.id

                if target_name and target_name in func_globals:
                    child = func_globals[target_name]
                    if getattr(child, "__wrapped__", None):
                        if target_name not in found_deps:
                            found_deps.add(target_name)
                            edges.append(f"  {parent_name} --> {target_name}")
                            queue.append(getattr(child, "__wrapped__"))

    return "graph LR\n" + "\n".join(edges)


# --- 3. THE CLASS WRAPPER ---


class StepWrapper:
    """
    A callable object that wraps the step function.
    Provides attribute access (.diagram, .schema) while maintaining callability.
    """

    def __init__(self, func: Callable):
        self._func = func
        self._sig = inspect.signature(func)
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        """
        Descriptor protocol support.
        Ensures that when a @step decorated method is accessed on an instance,
        we return a bound callable (via partial application) so 'self' is passed correctly.
        """
        if instance is None:
            return self

        # Create the bound method
        bound_method = functools.partial(self.__call__, instance)

        # PROXY the properties so they are accessible on the bound instance
        # This makes summarizer.__call__.schema work
        setattr(bound_method, "schema", self.schema)
        setattr(bound_method, "diagram", self.diagram)

        # Keep internal references for static analysis
        setattr(bound_method, "__wrapped__", self._func)

        return bound_method

    async def __call__(self, *args, **kwargs):
        # 1. Bind Args
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        unified_args = dict(bound.arguments)

        # Handle **kwargs
        var_kw = next(
            (
                p.name
                for p in self._sig.parameters.values()
                if p.kind == inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )
        if var_kw and var_kw in unified_args:
            extra = unified_args.pop(var_kw)
            if extra:
                unified_args.update(extra)

        # 2. Context
        start = time.time()
        token_args = context.args.set(unified_args)
        current_meta = {}
        token_meta = context.step_meta.set(current_meta)

        # Initialize defaults to prevent UnboundLocalError in finally block
        result = None
        status = "unknown"

        try:
            result = await self._func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "error"
            raise
        except BaseException as e:
            # Handle Cancellation/SystemExit
            result = f"<{type(e).__name__}>"
            status = "cancelled"
            raise
        finally:
            duration = time.time() - start
            trace = context.trace.get()
            if trace is not None:
                trace.append(
                    {
                        "step": self._func.__qualname__,
                        "inputs": unified_args,
                        "output": result,
                        "duration": round(duration, 4),
                        "status": status,
                        "metadata": current_meta,
                    }
                )
            context.args.reset(token_args)
            context.step_meta.reset(token_meta)

        return result

    @property
    def diagram(self) -> str:
        """Lazy-loaded Mermaid graph."""
        return _generate_hierarchy_graph(self._func)

    @property
    def schema(self) -> dict:
        """Lazy-loaded config schema."""
        return _static_scan_workflow(self._func)


def step(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    if not inspect.iscoroutinefunction(func):
        raise TypeError(
            f"@step requires async functions. {func.__name__} must be defined with `async def`."
        )

    # Return the class wrapper instead of a function closure
    return cast(Callable[P, Awaitable[R]], StepWrapper(func))
