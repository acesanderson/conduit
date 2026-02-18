from pydantic import BaseModel, Field
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardDatum,
    GeneratedSummary,
)


class EvalPoint(BaseModel):
    """
    The atomic unit of an evaluation run.
    Pairs the ground truth (GoldStandard) with the model's attempt (Generated)
    and the resulting metrics (Evaluation).
    """

    datum: GoldStandardDatum = Field(
        ...,
        description="The ground truth data, including source text and gold summary.",
    )
    generated: GeneratedSummary = Field(
        ..., description="The model's output and metadata from the harness."
    )
    evaluation: SummaryEvaluation | None = Field(
        default=None,
        description="The computed scores across semantic, faithfulness, and constraint domains.",
    )

    @property
    def source_id(self) -> str:
        return self.datum.entry.source_id

    @property
    def trace_id(self) -> str:
        return self.generated.trace_id
