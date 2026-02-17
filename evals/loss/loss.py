import abc
import asyncio
import re
import difflib
import numpy as np
from pydantic import BaseModel
from pydantic import Field
from dataclasses import dataclass
from sentence_transformers import CrossEncoder

# --- 1. Data Models ---


class ComponentScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    raw_value: float
    metadata: dict | None = None


class SummarizationEval(BaseModel):
    total_loss: float
    is_hard_fail: bool
    fail_reason: str | None = None
    metrics: dict[str, ComponentScore]
    tokens_actual: int
    tokens_target: int


# --- 2. Length Logic ---


@dataclass
class CompressionTier:
    max_input: int
    ratio: float
    min_tokens: int
    max_tokens: int


COMPRESSION_SCHEDULE = [
    CompressionTier(max_input=2000, ratio=0.15, min_tokens=300, max_tokens=400),
    CompressionTier(max_input=10000, ratio=0.10, min_tokens=400, max_tokens=1000),
    CompressionTier(max_input=40000, ratio=0.05, min_tokens=1000, max_tokens=2000),
]


def get_target_summary_length(input_tokens: int) -> int:
    for tier in COMPRESSION_SCHEDULE:
        if input_tokens <= tier.max_input:
            raw = int(input_tokens * tier.ratio)
            return max(tier.min_tokens, min(raw, tier.max_tokens))
    return 2500


# --- 3. Async Evaluators ---


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    async def compute(self, gold: BaseModel, gen_text: str) -> ComponentScore: ...


class FactEvaluator(BaseEvaluator):
    """Component A: Bidirectional Atomic Fact Recall (w=0.40)"""

    def __init__(self, model: CrossEncoder):
        self.model = model

    async def compute(self, gold: BaseModel, gen_text: str) -> ComponentScore:
        gen_sents = [
            s.strip() for s in re.split(r"(?<=[.!?]) +", gen_text) if s.strip()
        ]
        gold_facts = gold.key_facts

        # We run inference in threads to avoid blocking the event loop
        # Recall (Gold -> Gen)
        recall_hits = 0
        for fact in gold_facts:
            pairs = [[fact, s] for s in gen_sents]
            if not pairs:
                break
            logits = await asyncio.to_thread(self.model.predict, pairs)
            if any(l.argmax() == 1 for l in logits):
                recall_hits += 1

        # Precision (Gen -> Gold)
        precision_hits = 0
        for s in gen_sents:
            pairs = [[s, fact] for fact in gold_facts]
            logits = await asyncio.to_thread(self.model.predict, pairs)
            if any(l.argmax() == 1 for l in logits):
                precision_hits += 1

        r = recall_hits / len(gold_facts) if gold_facts else 1.0
        p = precision_hits / len(gen_sents) if gen_sents else 1.0
        f_score = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        return ComponentScore(
            score=f_score, raw_value=f_score, metadata={"p": p, "r": r}
        )


class EntityEvaluator(BaseEvaluator):
    """Component B: Entity Preservation (w=0.20)"""

    async def compute(self, gold: BaseModel, gen_text: str) -> ComponentScore:
        hits = 0
        gen_words = gen_text.split()

        # Fuzzy matching is fast but technically CPU-bound; running in thread for purity
        def sync_match():
            m = 0
            for ent in gold.entity_list:
                if difflib.get_close_matches(ent, gen_words, n=1, cutoff=0.9):
                    m += 1
            return m

        hits = await asyncio.to_thread(sync_match)
        score = hits / len(gold.entity_list) if gold.entity_list else 1.0
        return ComponentScore(score=score, raw_value=float(hits))


class FlowEvaluator(BaseEvaluator):
    """Component C: Structural Monotonicity (w=0.20)"""

    def __init__(self, model: CrossEncoder):
        self.model = model

    async def compute(self, gold: BaseModel, gen_text: str) -> ComponentScore:
        gen_sents = [
            s.strip() for s in re.split(r"(?<=[.!?]) +", gen_text) if s.strip()
        ]
        indices = []
        for point in gold.logical_outline:
            pairs = [[point, s] for s in gen_sents]
            if not pairs:
                continue
            logits = await asyncio.to_thread(self.model.predict, pairs)
            scores = logits[:, 1]
            if scores.max() > 0.5:
                indices.append(scores.argmax())

        coverage = (
            len(indices) / len(gold.logical_outline) if gold.logical_outline else 1.0
        )
        inversions = sum(
            1 for i in range(len(indices) - 1) if indices[i] > indices[i + 1]
        )
        order_score = (
            1.0 - (inversions / (len(indices) - 1)) if len(indices) > 1 else 1.0
        )

        final = (coverage * 0.7) + (order_score * 0.3)
        return ComponentScore(score=final, raw_value=coverage)


class SemanticEvaluator(BaseEvaluator):
    """Component D: Semantic Similarity (w=0.15)"""

    async def compute(self, gold: BaseModel, gen_text: str) -> ComponentScore:
        # Use your async helper library here
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        from headwater_api.classes import QuickEmbeddingRequest

        client = HeadwaterAsyncClient().embeddings

        # v_gold = await your_lib.embed(gold.summary)
        # v_gen = await your_lib.embed(gen_text)
        return ComponentScore(score=1.0, raw_value=1.0)


class LengthEvaluator:
    """Component E: Sublinear Length Penalty (w=0.05)"""

    async def compute(self, actual: int, target: int) -> ComponentScore:
        ratio = actual / target
        score = max(0.0, 1.0 - abs(1.0 - ratio))
        return ComponentScore(score=score, raw_value=float(actual))


# --- 4. Main Engine ---


class SummarizationLossEngine:
    def __init__(self, nli_model_name: str = "cross-encoder/nli-deberta-v3-small"):
        # Load model once on init
        model = CrossEncoder(nli_model_name)
        self.f_eval = FactEvaluator(model)
        self.e_eval = EntityEvaluator()
        self.o_eval = FlowEvaluator(model)
        self.s_eval = SemanticEvaluator()
        self.l_eval = LengthEvaluator()

        self.weights = {
            "facts": 0.40,
            "entities": 0.20,
            "flow": 0.20,
            "semantic": 0.15,
            "length": 0.05,
        }

    def _check_hard_fails(self, gen_text: str) -> tuple[bool, str | None]:
        if re.search(
            r"(This summary covers|The document states|In conclusion)", gen_text, re.I
        ):
            return True, "Meta-commentary detected"
        if not re.search(r"[.!?]$", gen_text.strip()):
            return True, "Truncated output"
        return False, None

    async def evaluate(self, datum: BaseModel, gen_text: str) -> SummarizationEval:
        is_fail, reason = self._check_hard_fails(gen_text)
        actual_len = len(gen_text.split())
        target_len = get_target_summary_length(datum.entry.token_count)

        if is_fail:
            return SummarizationEval(
                total_loss=1.0,
                is_hard_fail=True,
                fail_reason=reason,
                metrics={},
                tokens_actual=actual_len,
                tokens_target=target_len,
            )

        # Execute all metrics in parallel
        results = await asyncio.gather(
            self.f_eval.compute(datum.summary, gen_text),
            self.e_eval.compute(datum.summary, gen_text),
            self.o_eval.compute(datum.summary, gen_text),
            self.s_eval.compute(datum.summary, gen_text),
            self.l_eval.compute(actual_len, target_len),
        )

        scores = {
            "facts": results[0],
            "entities": results[1],
            "flow": results[2],
            "semantic": results[3],
            "length": results[4],
        }

        weighted_score = sum(self.weights[k] * scores[k].score for k in self.weights)

        return SummarizationEval(
            total_loss=round(1.0 - weighted_score, 4),
            is_hard_fail=False,
            metrics=scores,
            tokens_actual=actual_len,
            tokens_target=target_len,
        )
