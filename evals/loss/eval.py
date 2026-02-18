from __future__ import annotations
import asyncio
import numpy as np
from typing import Protocol
from pydantic import BaseModel, Field

from models import SummaryEvaluation, DomainScore
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardDatum,
    GeneratedSummary,
)


class EvalModule(Protocol):
    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore: ...


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class ConstraintModule:
    """
    Tier 1: Syntax Gate (The "Hard" Gate).
    Cheap, mechanical checks for instant rejection.
    """

    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore:
        from conduit.strategies.summarize.compression import is_within_threshold

        # 1. Length Adherence
        is_valid_len = is_within_threshold(
            datum.entry.token_count, generated.token_count
        )

        # 2. Format / Meta-commentary check
        forbidden = [
            "here is",
            "summary:",
            "the author",
            "this document",
            "i have summarized",
        ]
        # Check first 60 chars for chatty intros
        found_filler = [p for p in forbidden if p in generated.summary.lower()[:60]]

        score = 1.0 if (is_valid_len and not found_filler) else 0.0

        reasoning = "Passed Syntax Gate" if score == 1.0 else ""
        if not is_valid_len:
            reasoning += "Failed Length Adherence. "
        if found_filler:
            reasoning += f"Detected conversational filler: {found_filler}."

        return DomainScore(
            score=score,
            reasoning=reasoning.strip(),
            metadata={"is_valid_len": is_valid_len, "found_filler": found_filler},
        )


class SemanticRecallModule:
    """
    Tier 2: Semantics Gate (The "Drift" Gate).
    Naive embedding distance check to catch models that 'lost the plot'.
    """

    async def compute(
        self, generated: GeneratedSummary, datum: GoldStandardDatum
    ) -> DomainScore:
        from headwater_client.client.headwater_client_async import HeadwaterClientAsync
        from headwater_api.classes import EmbeddingsRequest, ChromaBatch

        client = HeadwaterClientAsync().embeddings

        batch = ChromaBatch(ids=[generated.trace_id], documents=[generated.summary])
        request = EmbeddingsRequest(model="google/embeddinggemma-300m", batch=batch)

        response = await client.create_embeddings(request)
        if not response.embeddings:
            return DomainScore(score=0.0, reasoning="Embedding generation failed.")

        similarity = cosine_similarity(
            response.embeddings[0], datum.summary.summary_embeddings
        )

        return DomainScore(
            score=round(float(similarity), 4),
            reasoning=f"Semantic similarity: {similarity:.4f}",
            metadata={"model": "embeddinggemma-300m"},
        )


class SummarizationEvaluator:
    """
    The Tri-Gate Orchestrator.
    Implements cascading gates: Syntax (Hard) -> Semantics (Drift) -> Cognition (Judge).
    """

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
        # --- TIER 1: SYNTAX ---
        c_score = await self.constraint.compute(generated, datum)
        if c_score.score == 0.0:
            return self._abort(datum, generated, tier="Syntax", result=c_score)

        # --- TIER 2: SEMANTICS ---
        s_score = await self.semantic.compute(generated, datum)
        if s_score.score < self.drift_threshold:
            return self._abort(datum, generated, tier="Semantic", result=s_score)

        # --- TIER 3: COGNITION ---
        j_score = await self.judge.compute(generated, datum)

        # FINAL LOSS (Accumulated Quality)
        # Weights: Judge (50%), Semantic (30%), Constraint (20%)
        total = (c_score.score * 0.2) + (s_score.score * 0.3) + (j_score.score * 0.5)

        return SummaryEvaluation(
            source_id=datum.entry.source_id,
            trace_id=generated.trace_id,
            semantic_recall=s_score,
            faithfulness=j_score,
            constraint_alignment=c_score,
            total_score=round(total, 4),
        )

    def _abort(
        self,
        datum: GoldStandardDatum,
        gen: GeneratedSummary,
        tier: str,
        result: DomainScore,
    ) -> SummaryEvaluation:
        """Standardized short-circuit for gate failures."""
        return SummaryEvaluation(
            source_id=datum.entry.source_id,
            trace_id=gen.trace_id,
            semantic_recall=result if tier == "Semantic" else DomainScore(score=0.0),
            faithfulness=DomainScore(score=0.0),
            constraint_alignment=result if tier == "Syntax" else DomainScore(score=1.0),
            total_score=0.0,
            metadata={"abort_tier": tier, "abort_reason": result.reasoning},
        )
