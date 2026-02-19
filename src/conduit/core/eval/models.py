from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pydantic import BaseModel, Field
from datetime import datetime

if TYPE_CHECKING:
    pass


class Document(BaseModel):
    """Raw workflow input. Content + metadata only — no gold, no eval concerns."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # metadata examples: token_count, source, title, etc.


class GoldSummary(BaseModel):
    """Reference output for a document. Stands alone; not tied to any workflow."""

    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldDatum(BaseModel):
    """A paired document + reference output. The unit of a golden dataset."""

    document: Document
    gold_summary: GoldSummary


class WorkflowInput(BaseModel):
    """What the harness receives. Harness knows nothing beyond this."""

    document: Document
    config: dict[str, Any]


class WorkflowOutput(BaseModel):
    """What the harness returns. No knowledge of gold or eval concerns."""

    output: Any
    trace: dict[str, Any] = Field(default_factory=dict)
    experiment_config_id: str | None = None  # set by Experiment after persisting config


class EvalInput(BaseModel):
    """
    What the evaluator receives. Assembles run artifacts for scoring.
    gold_summary is optional — evaluators declare whether they require it.
    """

    workflow_input: WorkflowInput
    workflow_output: WorkflowOutput
    gold_summary: GoldSummary | None = None


class EvalOutput(BaseModel):
    """
    The result of an evaluation run. Flexible blob — schema enforced by Pydantic,
    not the database. Common fields include score: float and passed: bool,
    but evaluators may include arbitrary rubric fields in `details`.

    Examples:
        score: float          # e.g. 0.0–1.0 normalized score
        passed: bool          # e.g. score >= threshold
        details: dict         # rubric breakdown, per-criterion scores, LLM reasoning, etc.
    """

    experiment_config_id: str | None = None
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = Field(default_factory=dict)
