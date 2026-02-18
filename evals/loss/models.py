from pydantic import BaseModel, Field
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardDatum,
    GeneratedSummary,
)


class DomainScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str | None = None
    metadata: dict | None = None


class SummaryEvaluation(BaseModel):
    source_id: str
    semantic_recall: DomainScore  # Accuracy/Embeddings
    faithfulness: DomainScore  # Hallucination check
    constraint_alignment: DomainScore  # Length/Format
    total_score: float = Field(..., ge=0.0, le=1.0)


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
