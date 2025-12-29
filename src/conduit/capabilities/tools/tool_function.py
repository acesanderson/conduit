from __future__ import annotations
import inspect
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
from typing import Annotated
from typing import Protocol
from typing import get_args
from typing import get_origin
from typing import runtime_checkable


@runtime_checkable
class ToolFunction(Protocol):
    """
    A function eligible for Tool generation.

    Hard requirements:
    - Stable name (not a lambda; has __name__)
    - Non-empty docstring
    - No *args or **kwargs
    - Every parameter MUST be typing.Annotated[T, <non-empty str description>, ...]
      (Type must be present as the first Annotated arg; description must be a str metadata item.)
    """

    __name__: str
    __doc__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ToolFunctionError:
    code: str
    message: str
    details: dict[str, Any] | None = None


def _clean_doc(doc: str | None) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).strip()


def _is_annotated(tp: Any) -> bool:
    return get_origin(tp) is Annotated


def _extract_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """
    Returns (base_type, metadata_list). If not Annotated, base_type is tp and metadata is [].
    """
    if get_origin(tp) is Annotated:
        base, *meta = get_args(tp)
        return base, list(meta)
    return tp, []


def _extract_description(meta: list[Any]) -> str | None:
    """
    Canonical rule: first non-empty string metadata item is the description.
    """
    for m in meta:
        if isinstance(m, str) and m.strip():
            return m.strip()
    return None


def validate_tool_function(func: Callable[..., Any]) -> list[ToolFunctionError]:
    errors: list[ToolFunctionError] = []

    # Must be introspectable
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return [
            ToolFunctionError(
                "not_introspectable", "Object has no inspectable signature."
            )
        ]

    # Stable name required
    name = getattr(func, "__name__", "")
    if not name or name == "<lambda>":
        errors.append(
            ToolFunctionError(
                code="unstable_name",
                message="Function must have a stable __name__ (lambdas are not allowed).",
                details={"name": name},
            )
        )

    # Docstring required
    doc = _clean_doc(getattr(func, "__doc__", None))
    if not doc:
        errors.append(
            ToolFunctionError(
                code="missing_docstring",
                message="Function must have a non-empty docstring.",
            )
        )

    # Parameters: no *args/**kwargs; every param must be Annotated with a description
    for param_name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            errors.append(
                ToolFunctionError(
                    code="var_positional_not_supported",
                    message="*args is not supported for tool functions.",
                    details={"param": param_name},
                )
            )
            continue

        if param.kind is inspect.Parameter.VAR_KEYWORD:
            errors.append(
                ToolFunctionError(
                    code="var_keyword_not_supported",
                    message="**kwargs is not supported for tool functions.",
                    details={"param": param_name},
                )
            )
            continue

        ann = param.annotation
        if ann is inspect._empty:
            errors.append(
                ToolFunctionError(
                    code="missing_annotation",
                    message="Every parameter must be annotated using typing.Annotated[...].",
                    details={"param": param_name},
                )
            )
            continue

        if not _is_annotated(ann):
            errors.append(
                ToolFunctionError(
                    code="annotated_required",
                    message="Every parameter must use typing.Annotated[T, 'description', ...].",
                    details={"param": param_name, "annotation": repr(ann)},
                )
            )
            continue

        base_tp, meta = _extract_annotated(ann)

        # Base type sanity (must exist)
        if base_tp is None or base_tp is inspect._empty:
            errors.append(
                ToolFunctionError(
                    code="annotated_missing_base_type",
                    message="Annotated parameter must include a base type as the first argument.",
                    details={"param": param_name},
                )
            )

        # Description required
        desc = _extract_description(meta)
        if not desc:
            errors.append(
                ToolFunctionError(
                    code="missing_param_description",
                    message="Annotated parameter must include a non-empty string description.",
                    details={"param": param_name},
                )
            )

    return errors


def assert_tool_function(func: Callable[..., Any]) -> ToolFunction:
    errs = validate_tool_function(func)
    if errs:
        lines = []
        for e in errs:
            tail = f" {e.details}" if e.details else ""
            lines.append(f"- {e.code}: {e.message}{tail}")
        raise TypeError("Invalid ToolFunction:\n" + "\n".join(lines))
    return func  # type: ignore[return-value]
