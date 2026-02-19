from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from conduit.core.eval.models import EvalInput, EvalOutput

if TYPE_CHECKING:
    pass


class Evaluator(ABC):
    """
    Base class for all evaluators. Subclasses implement evaluate().

    An evaluator is a pure function over EvalInput -> EvalOutput.
    It has no knowledge of workflow config, harness internals, or persistence.
    Optionally accepts an eval_config for parameterizing the evaluator itself
    (e.g. judge model, scoring thresholds, rubric weights).
    """

    def __init__(self, eval_config: dict[str, Any] | None = None) -> None:
        self.eval_config: dict[str, Any] = eval_config or {}

    @abstractmethod
    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        """
        Score a single workflow run.

        Args:
            eval_input: Assembled input containing workflow_input, workflow_output,
                        and optionally a gold_summary.

        Returns:
            EvalOutput with scores and rubric details.
        """
        ...

    def requires_gold(self) -> bool:
        """
        Declare whether this evaluator requires a gold_summary.
        Override in subclasses that need a reference output.
        Default is False (reference-free evaluation).
        """
        return False
