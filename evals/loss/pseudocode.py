import abc
import numpy
from pydantic import BaseModel
from pydantic import Field
from dataclasses import dataclass


# --- Models ---
class ComponentScore(BaseModel):
    score: float = Field(..., description="Normalized 0.0 to 1.0 (1.0 is perfect)")
    raw_value: float
    metadata: dict | None = None


class SummarizationEval(BaseModel):
    total_loss: float
    is_hard_fail: bool
    fail_reason: str | None = None
    metrics: dict[str, ComponentScore]
    tokens_actual: int
    tokens_target: int


# --- Component Interfaces (DI) ---
class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def compute(self, gold: GoldStandardSummary, gen_text: str) -> ComponentScore: ...


class FactEvaluator(BaseEvaluator):
    """Calculates Fact Recall and Precision using NLI."""

    def compute(self, gold: GoldStandardSummary, gen_text: str) -> ComponentScore: ...


class EntityEvaluator(BaseEvaluator):
    """Checks for Entity Preservation with Contextual Fuzzy Matching."""

    def compute(self, gold: GoldStandardSummary, gen_text: str) -> ComponentScore: ...


class FlowEvaluator(BaseEvaluator):
    """Validates Structural Monotonicity based on the logical outline."""

    def compute(self, gold: GoldStandardSummary, gen_text: str) -> ComponentScore: ...


class SemanticEvaluator(BaseEvaluator):
    """Calculates Cosine Similarity in vector space."""

    def compute(self, gold: GoldStandardSummary, gen_text: str) -> ComponentScore: ...


class LengthEvaluator:
    """Calculates the Sublinear Length Penalty."""

    def compute(self, actual_len: int, target_len: int) -> ComponentScore: ...


# --- Core Engine ---


class SummarizationLossEngine:
    def __init__(
        self,
        fact_eval: FactEvaluator,
        entity_eval: EntityEvaluator,
        flow_eval: FlowEvaluator,
        semantic_eval: SemanticEvaluator,
        length_eval: LengthEvaluator,
    ):
        self.evaluators = {
            "facts": fact_eval,
            "entities": entity_eval,
            "flow": flow_eval,
            "semantic": semantic_eval,
            "length": length_eval,
        }
        self.weights = {
            "facts": 0.40,
            "entities": 0.20,
            "flow": 0.20,
            "semantic": 0.15,
            "length": 0.05,
        }

    def evaluate(self, datum: GoldStandardDatum, gen_text: str) -> SummarizationEval:
        """
        Executes the full evaluation pipeline for a single summary.
        """
        # 1. Check Hard Fails
        is_fail, reason = self._check_hard_fails(gen_text)
        if is_fail:
            return self._build_fail_response(reason, gen_text, datum)

        # 2. Get Length Targets
        target_len = get_target_summary_length(datum.entry.token_count)
        actual_len = len(gen_text.split())  # Simplified for now

        # 3. Compute Individual Components
        scores = {
            "facts": self.evaluators["facts"].compute(datum.summary, gen_text),
            "entities": self.evaluators["entities"].compute(datum.summary, gen_text),
            "flow": self.evaluators["flow"].compute(datum.summary, gen_text),
            "semantic": self.evaluators["semantic"].compute(datum.summary, gen_text),
            "length": self.evaluators["length"].compute(actual_len, target_len),
        }

        # 4. Aggregate Loss
        total_loss = self._calculate_total_loss(scores)

        return SummarizationEval(
            total_loss=total_loss,
            is_hard_fail=False,
            metrics=scores,
            tokens_actual=actual_len,
            tokens_target=target_len,
        )

    def _check_hard_fails(self, gen_text: str) -> tuple[bool, str | None]:
        """Regex-based checks for meta-commentary and truncation."""
        ...

    def _calculate_total_loss(self, scores: dict[str, ComponentScore]) -> float:
        """Weighted sum inverted: 1.0 - Sum(Weight * Score)."""
        weighted_score = sum(self.weights[k] * scores[k].score for k in self.weights)
        return 1.0 - weighted_score

    def _build_fail_response(
        self, reason: str, gen_text: str, datum: GoldStandardDatum
    ) -> SummarizationEval:
        """Returns a max-loss object for hard failures."""
        ...
