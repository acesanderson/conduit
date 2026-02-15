from typing import Any, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from typing import Any
from conduit.core.workflow.step import StepWrapper


@runtime_checkable
class Step(Protocol):
    """
    A unit of work. Must be decorated with @step.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class Strategy(ABC):
    """
    Abstract base class for workflow strategies that enforces telemetry-enabled async execution.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # FIX: Only validate if the class is NOT an Abstract Base Class
        if ABC not in cls.__bases__ and "__call__" in cls.__dict__:
            from conduit.core.workflow.step import StepWrapper

            if not isinstance(cls.__call__, StepWrapper):
                raise TypeError(
                    f"Concrete Strategy '{cls.__name__}' must decorate __call__ with @step to enable telemetry."
                )

    @property
    def schema(self) -> dict:
        return getattr(self.__call__, "schema", {})

    @property
    def diagram(self) -> str:
        return getattr(self.__call__, "diagram", "")

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Requirement: Must be an async @step."""
        ...


@runtime_checkable
class Workflow(Protocol):
    """
    The orchestrator script.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
