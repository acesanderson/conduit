from __future__ import annotations
import numpy as np
from typing import Protocol
from pydantic import BaseModel, Field
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardDatum,
    GeneratedSummary,
)
from run import run_config, CONFIG_DICT, EXAMPLE_TEXT
import asyncio


# --- Data Models ---
class DomainScore(BaseModel):
    score: float
    reasoning: str
    metadata: dict = Field(default_factory=dict)


class SummaryEvaluation(BaseModel):
    source_id: str
    semantic_recall: DomainScore  # Accuracy/Embeddings
    faithfulness: DomainScore  # Hallucination check
    constraint_alignment: DomainScore  # Length/Format
    total_score: float = Field(..., ge=0.0, le=1.0)


class JudgeRubric(BaseModel):
    """
    Structured audit for Tier 3 Cognition Gate.
    """

    total_atomic_facts: int = Field(
        ..., description="Count of independent statements in the Gold Standard."
    )
    supported_facts: int = Field(
        ..., description="Count of Gold facts accurately present in the Candidate."
    )
    hallucinations: list[str] = Field(
        default_factory=list, description="Claims in Candidate not supported by Gold."
    )
    reasoning: str = Field(
        ..., description="Technical justification for the audit results."
    )


# --- Protocols & Utils ---
class EvalModule(Protocol):
    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore: ...


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- Modules ---
class ConstraintModule:
    """
    Tier 1: Syntax Gate. 20% of total score.
    Checks length adherence and conversational filler.
    """

    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore:
        from conduit.strategies.summarize.compression import is_within_threshold

        is_valid_len = is_within_threshold(
            datum.entry.token_count, generated.token_count
        )

        forbidden = ["here is", "summary:", "the author", "this document"]
        found_filler = [p for p in forbidden if p in generated.summary.lower()[:60]]

        score = 1.0 if (is_valid_len and not found_filler) else 0.0
        reasoning = "Passed Syntax Gate" if score == 1.0 else "Failed Syntax Gate."

        return DomainScore(
            score=score,
            reasoning=reasoning,
            metadata={"is_valid_len": is_valid_len, "found_filler": found_filler},
        )


class SemanticRecallModule:
    """
    Tier 2: Semantics Gate. 10% of total score.
    Catch-all for significant topic drift.
    """

    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore:
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        from headwater_api.classes import EmbeddingsRequest, ChromaBatch

        client = HeadwaterAsyncClient().embeddings
        batch = ChromaBatch(ids=[generated.trace_id], documents=[generated.summary])
        request = EmbeddingsRequest(model="google/embeddinggemma-300m", batch=batch)

        response = await client.generate_embeddings(request)
        if not response.embeddings:
            return DomainScore(score=0.0, reasoning="Embedding generation failed.")

        similarity = cosine_similarity(
            response.embeddings[0], datum.summary.summary_embeddings
        )

        return DomainScore(
            score=round(float(similarity), 4),
            reasoning=f"Similarity: {similarity:.4f}",
            metadata={"model": "embeddinggemma-300m"},
        )


class CognitionModule:
    """
    Tier 3: Cognition Gate. 70% of total score.
    Uses LLM-as-a-Judge to perform Atomic Fact Recall.
    """

    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore:
        from conduit.async_ import (
            ConduitAsync,
            Prompt,
            GenerationParams,
            ConduitOptions,
            Verbosity,
        )
        from conduit.config import settings

        params = GenerationParams(
            model="gemini3",
            response_model=JudgeRubric,
            output_type="structured_response",
        )
        options = ConduitOptions(
            project_name="SummarizationEvaluation",
            cache=settings.default_cache("SummarizationEvaluation"),
            verbosity=Verbosity.PROGRESS,
        )

        prompt = """
        Analyze the facts in the <candidate> against the <gold_standard>.
        
        <gold_standard>
        {{ gold_summary }}
        </gold_standard>
        
        <candidate>
        {{ candidate_summary }}
        </candidate>
        
        Deconstruct the gold standard into atomic facts and verify if the candidate contains them. 
        Identify any hallucinations.
        """

        conduit = ConduitAsync(prompt=Prompt(prompt))

        input_variables = {
            "gold_summary": datum.summary.summary,
            "candidate_summary": generated.summary,
        }

        response = await conduit.run(
            input_variables=input_variables, params=params, options=options
        )

        rubric: JudgeRubric = response.last.parsed
        total = max(rubric.total_atomic_facts, 1)
        recall = rubric.supported_facts / total

        # Penalty: -0.2 per hallucination
        penalty = len(rubric.hallucinations) * 0.2
        final_score = max(0.0, recall - penalty)

        return DomainScore(
            score=round(final_score, 4),
            reasoning=rubric.reasoning,
            metadata={
                "total_facts": rubric.total_atomic_facts,
                "supported": rubric.supported_facts,
                "hallucination_count": len(rubric.hallucinations),
            },
        )


# --- Orchestrator ---
class SummarizationEvaluator:
    def __init__(
        self,
        constraint_engine: EvalModule,
        semantic_engine: EvalModule,
        judge_engine: EvalModule,
        drift_threshold: float = 0.6,
    ):
        self.constraint = constraint_engine
        self.semantic = semantic_engine
        self.judge = judge_engine
        self.drift_threshold = drift_threshold

    async def evaluate(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> SummaryEvaluation:
        # Tier 1: Syntax
        c_score = await self.constraint.compute(generated, datum)
        if c_score.score == 0.0:
            return self._abort(datum, generated, "Syntax", c_score)

        # Tier 2: Semantic
        s_score = await self.semantic.compute(generated, datum)
        if s_score.score < self.drift_threshold:
            return self._abort(datum, generated, "Semantic", s_score)

        # Tier 3: Cognition
        j_score = await self.judge.compute(generated, datum)

        # Weighted Loss Function
        # 70% Judge / 20% Syntax / 10% Semantic
        total = (j_score.score * 0.7) + (c_score.score * 0.2) + (s_score.score * 0.1)

        return SummaryEvaluation(
            source_id=datum.entry.source_id,
            trace_id=generated.trace_id,
            semantic_recall=s_score,
            faithfulness=j_score,
            constraint_alignment=c_score,
            total_score=round(total, 4),
        )

    def _abort(self, datum, gen, tier, result) -> SummaryEvaluation:
        return SummaryEvaluation(
            source_id=datum.entry.source_id,
            trace_id=gen.trace_id,
            semantic_recall=result
            if tier == "Semantic"
            else DomainScore(score=0.0, reasoning="Skipped"),
            faithfulness=DomainScore(score=0.0, reasoning="Gate Abort"),
            constraint_alignment=result
            if tier == "Syntax"
            else DomainScore(score=1.0, reasoning="Passed"),
            total_score=0.0,
            metadata={"abort_tier": tier, "abort_reason": result.reasoning},
        )


async def eval_run(
    config: dict, datums: list[GoldStandardDatum]
) -> list[SummaryEvaluation]:
    evaluator = SummarizationEvaluator(
        constraint_engine=ConstraintModule(),
        semantic_engine=SemanticRecallModule(),
        judge_engine=CognitionModule(),
    )

    coroutines = []

    for datum in datums:
        generated_summary = await run_config(config, text=datum.entry.text)
        coroutines.append(evaluator.evaluate(generated_summary, datum))
    evaluations = await asyncio.gather(*coroutines)
    return evaluations


if __name__ == "__main__":
    from conduit.strategies.summarize.datasets.load_datasets import load_datums

    dataset = load_datums()
    evals = asyncio.run(eval_run(CONFIG_DICT, dataset[:5]))
