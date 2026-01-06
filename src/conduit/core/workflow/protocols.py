from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Step(Protocol):
    """
    A unit of work. Must be decorated with @step.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class Strategy(Step, Protocol):
    """
    A Step that adheres to a strict interface.
    Inherit this for plugins like Summarizer, Researcher, etc.
    """

    pass


@runtime_checkable
class Workflow(Protocol):
    """
    The orchestrator script.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
