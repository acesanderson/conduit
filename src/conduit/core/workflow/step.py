from __future__ import annotations
import ast
import functools
import inspect
import textwrap
import time
from collections import deque
from collections.abc import Callable, Awaitable
from typing import Any, TypeVar, ParamSpec, cast, TYPE_CHECKING

from conduit.core.workflow.context import context

if TYPE_CHECKING:
    from conduit.core.workflow.protocols import Workflow

P = ParamSpec("P")
R = TypeVar("R")


def resolve_param(
    key: str,
    default: Any = None,
    overrides: dict | None = None,
    scope: str | None = None,
) -> Any:
    if scope is None:
        try:
            stack = inspect.stack()
            caller_frame = None
            for frame_info in stack[1:]:
                if frame_info.function not in ("resolve_param", "get_param"):
                    caller_frame = frame_info.frame
                    break
            if caller_frame and "self" in caller_frame.f_locals:
                scope = caller_frame.f_locals["self"].__class__.__name__
            elif caller_frame:
                scope = caller_frame.f_code.co_name
        except Exception:
            scope = "unknown"

    runtime_args = overrides or context.args.get() or {}
    if key in runtime_args:
        return runtime_args[key]

    cfg = context.config.get()
    if scope and f"{scope}.{key}" in cfg:
        return cfg[f"{scope}.{key}"]
    if key in cfg:
        return cfg[key]

    if context.use_defaults.get() is True:
        return default

    raise KeyError(
        f"Required parameter '{key}' not found in configuration (scope: {scope})."
    )


def get_param(key: str, default: Any = None, scope: str | None = None) -> Any:
    return resolve_param(key, default, None, scope)


def add_metadata(key: str, value: Any) -> None:
    meta = context.step_meta.get()
    if meta is not None:
        meta[key] = value


def _static_scan_workflow(root_func) -> dict[str, dict]:
    param_requirements = {}
    visited = set()
    root = getattr(root_func, "__wrapped__", root_func)
    queue = deque([root])
    visited.add(root)

    while queue:
        current_func = queue.popleft()
        try:
            src = textwrap.dedent(inspect.getsource(current_func))
            tree = ast.parse(src)
            scope = (
                current_func.__qualname__.split(".")[0]
                if "." in current_func.__qualname__
                else current_func.__name__
            )

            func_scope = getattr(current_func, "__globals__", {}).copy()
            if "." in current_func.__qualname__:
                module = inspect.getmodule(current_func)
                cls = getattr(module, scope, None)
                if cls:
                    func_scope.update(cls.__dict__)
        except (OSError, TypeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_id = ""
                if isinstance(node.func, ast.Name):
                    call_id = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_id = node.func.attr

                if call_id in ("resolve_param", "get_param"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        key = node.args[0].value
                        has_code_default = len(node.args) > 1 or any(
                            kw.arg == "default" for kw in node.keywords
                        )
                        logical_name = f"{scope}.{key}"
                        param_requirements[logical_name] = {
                            "keys": [logical_name, key],
                            "has_code_default": has_code_default,
                        }

                target_obj = func_scope.get(call_id)
                if target_obj:
                    underlying = getattr(target_obj, "__wrapped__", target_obj)
                    if callable(underlying) and underlying not in visited:
                        visited.add(underlying)
                        queue.append(underlying)
    return param_requirements


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
            src = textwrap.dedent(inspect.getsource(current_func))
            tree = ast.parse(src)
            func_globals = getattr(current_func, "__globals__", {})
        except (OSError, TypeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target_name = None
                if isinstance(node.func, ast.Name):
                    target_name = node.func.id
                elif isinstance(node.func, ast.Call) and isinstance(
                    node.func.func, ast.Name
                ):
                    target_name = node.func.func.id
                elif isinstance(node.func, ast.Attribute):
                    target_name = node.func.attr
                if target_name and target_name in func_globals:
                    child = func_globals[target_name]
                    if hasattr(child, "__wrapped__") or hasattr(child, "schema"):
                        edges.append(f"  {parent_name} --> {target_name}")
                        underlying = getattr(child, "__wrapped__", child)
                        if inspect.isclass(underlying) and hasattr(
                            underlying, "__call__"
                        ):
                            underlying = getattr(
                                underlying.__call__, "__wrapped__", underlying.__call__
                            )
                        queue.append(underlying)
    return "graph LR\n" + "\n".join(edges)


class StepWrapper:
    def __init__(self, func: Callable):
        self._func = func
        self._sig = inspect.signature(func)
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        bound_method = functools.partial(self.__call__, instance)
        setattr(bound_method, "schema", self.schema)
        setattr(bound_method, "diagram", self.diagram)
        setattr(bound_method, "__wrapped__", self._func)
        return bound_method

    async def __call__(self, *args, **kwargs):
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        unified_args = {
            k: v for k, v in bound.arguments.items() if k not in ("self", "cls")
        }
        start = time.time()
        token_args = context.args.set(unified_args)
        current_meta = {}
        token_meta = context.step_meta.set(current_meta)
        result, status = None, "unknown"
        try:
            result = await self._func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result, status = str(e), "error"
            raise
        finally:
            duration = round(time.time() - start, 4)
            trace = context.trace.get()
            if trace is not None:
                trace.append(
                    {
                        "step": self._func.__qualname__,
                        "inputs": unified_args,
                        "output": result,
                        "duration": duration,
                        "status": status,
                        "metadata": current_meta,
                    }
                )
            context.args.reset(token_args)
            context.step_meta.reset(token_meta)
        return result

    @property
    def diagram(self) -> str:
        return _generate_hierarchy_graph(self._func)

    @property
    def schema(self) -> dict:
        return _static_scan_workflow(self._func)


def step(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"@step requires async functions: {func.__name__}")
    return cast(Callable[P, Awaitable[R]], StepWrapper(func))
