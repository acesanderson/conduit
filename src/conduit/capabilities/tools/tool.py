from __future__ import annotations
import inspect
from typing import Annotated, Any, Literal, get_args, get_origin, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from conduit.capabilities.tools.tool_function import ToolFunction
from conduit.capabilities.tools.tool_function import validate_tool_function

if TYPE_CHECKING:
    from conduit.capabilities.tools.registry import ToolRegistry


JsonType = Literal["string", "number", "boolean", "integer", "object", "array"]


class ToolCallError(Exception):
    """Custom exception for ToolCall errors."""


class Property(BaseModel):
    type: JsonType
    description: str | None = None
    enum: list[str] | None = None
    items: Property | None = None  # only for arrays


class ObjectSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Property]
    required: list[str] = []
    additional_properties: bool = Field(default=False, alias="additionalProperties")


def _clean_doc(doc: str | None) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).strip()


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is None:
        return False
    if origin is type(None):  # noqa: E721
        return True
    if origin is getattr(__import__("typing"), "Union"):
        return any(a is type(None) for a in get_args(tp))  # noqa: E721
    return False


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is getattr(__import__("typing"), "Union"):
        args = tuple(a for a in get_args(tp) if a is not type(None))  # noqa: E721
        if len(args) == 1:
            return args[0]
        return origin[args]  # type: ignore[index]
    return tp


def _extract_annotated(tp: Any) -> tuple[Any, str]:
    if get_origin(tp) is not Annotated:
        raise TypeError("Parameter annotation must be typing.Annotated[...]")
    base, *meta = get_args(tp)
    desc = next((m.strip() for m in meta if isinstance(m, str) and m.strip()), None)
    if desc is None:
        raise TypeError(
            "Annotated parameter must include a non-empty string description."
        )
    return base, desc


def _python_type_to_property(tp: Any) -> Property:
    """
    Minimal mapping from Python type annotations to canonical JSON types.
    Assumes `tp` is the *base type* (i.e., Annotated already unwrapped).
    """
    if tp is inspect._empty:
        return Property(type="string")

    if _is_optional(tp):
        tp = _strip_optional(tp)

    origin = get_origin(tp)

    if origin in (list, tuple):
        args = get_args(tp)
        item_tp = args[0] if args else Any
        return Property(type="array", items=_python_type_to_property(item_tp))

    if origin is dict:
        return Property(type="object")

    if tp is str:
        return Property(type="string")
    if tp is int:
        return Property(type="integer")
    if tp is float:
        return Property(type="number")
    if tp is bool:
        return Property(type="boolean")

    # Fallback for unknown / complex types (including nested models)
    return Property(type="object")


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["function"] = "function"
    name: str
    description: str
    input_schema: ObjectSchema

    # Callable attribute
    func: ToolFunction = Field(exclude=True)

    # Factory method
    @classmethod
    def from_function(cls, func: ToolFunction) -> Tool:
        """
        Generate a Tool from a function.

        Strict requirements are enforced by validate_tool_function:
        - stable name required
        - docstring required
        - Annotated required for each parameter (with per-param description)
        """
        # Validate the function first
        errors = validate_tool_function(func)
        if errors:
            lines = []
            for e in errors:
                tail = f" {e.details}" if getattr(e, "details", None) else ""
                lines.append(f"- {e.code}: {e.message}{tail}")
            raise TypeError("Invalid ToolFunction:\n" + "\n".join(lines))

        sig = inspect.signature(func)

        properties: dict[str, Property] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    f"Tool functions cannot use *args/**kwargs: {param_name}"
                )

            base_tp, desc = _extract_annotated(param.annotation)
            prop = _python_type_to_property(base_tp)
            prop.description = desc
            properties[param_name] = prop

            is_required = (param.default is inspect._empty) and (
                not _is_optional(base_tp)
            )
            if is_required:
                required.append(param_name)

        schema = ObjectSchema(
            properties=properties,
            required=required,
            additional_properties=False,
        )

        description = _clean_doc(getattr(func, "__doc__", None))
        if not description:
            raise TypeError("ToolFunction docstring must be non-empty.")

        name = getattr(func, "__name__", "")
        if not name or name == "<lambda>":
            raise TypeError(
                "ToolFunction name must be stable and non-empty (lambdas not allowed)."
            )

        return cls(
            name=name,
            description=description,
            input_schema=schema,
            func=func,
        )

    def register(self, registry: ToolRegistry):
        registry.register(self)
