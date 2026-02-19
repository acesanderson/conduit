from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from conduit.core.eval.evaluator import Evaluator
from conduit.core.eval.models import (
    EvalInput,
    EvalOutput,
    GoldDatum,
    WorkflowInput,
    WorkflowOutput,
)
from conduit.core.workflow.harness import ConduitHarness

if TYPE_CHECKING:
    from collections.abc import Callable


class ExperimentConfig(BaseModel):
    """
    A single fully-resolved, labeled config dict with a stable identity.
    The ID is a hash of the config contents — identical configs produce the same ID.
    Attach a label for human-readable identification in results.
    """

    label: str
    config: dict[str, Any]
    id: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            payload = json.dumps(self.config, sort_keys=True)
            self.id = hashlib.sha256(payload.encode()).hexdigest()[:12]


class ExperimentResult(BaseModel):
    """The paired output of a single (GoldDatum, ExperimentConfig) run."""

    experiment_config_id: str
    workflow_output: WorkflowOutput
    eval_output: EvalOutput


class Experiment:
    """
    Owns the cartesian product of GoldDatum x ExperimentConfig.
    Fans out harness runs, assembles EvalInputs, dispatches evaluators,
    and collects results. Persistence is out of scope here.
    """

    def __init__(
        self,
        workflow: Callable,
        evaluators: list[Evaluator],
        gold_data: list[GoldDatum],
        configs: list[ExperimentConfig],
    ) -> None: ...

    async def run(self) -> list[ExperimentResult]:
        """
        Execute the full experiment matrix.
        Returns one ExperimentResult per (GoldDatum, ExperimentConfig) pair.
        """
        ...

    async def _run_single(
        self,
        datum: GoldDatum,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        """
        Run one cell of the matrix: harness run + all evaluators.
        Assembles WorkflowInput, runs harness, builds EvalInput per evaluator.
        """
        ...
