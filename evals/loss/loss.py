import asyncio
import re
import numpy
from dataclasses import dataclass
from headwater_api.classes import QuickEmbeddingRequest


@dataclass
class CompressionTier:
    max_input: int
    ratio: float
    min_tokens: int
    max_tokens: int


@dataclass
class EvalComponent:
    score: float
    weight: float
    raw_metrics: dict


class SummaryEval:
    def __init__(self, total_loss, is_hard_fail, fail_reason=None, components=None):
        self.total_loss = total_loss
        self.is_hard_fail = is_hard_fail
        self.fail_reason = fail_reason
        self.components = components or {}


class MultiFactorLoss:
    def __init__(
        self,
        hw_client,
        gliner_model,
        nli_model,
        compression_schedule: list,
        entity_threshold: float = 0.85,
    ):
        self.hw_client = hw_client
        self.gliner = gliner_model
        self.nli = nli_model
        self.schedule = compression_schedule
        self.entity_threshold = entity_threshold

    async def _get_embedding(self, text: str) -> list[float]:
        req = QuickEmbeddingRequest(query=text, model="google/embedding-gemma-300m")
        resp = await self.hw_client.quick_embedding(req)
        return resp.embeddings

    def _get_target_length(self, input_tokens: int) -> int:
        for tier in self.schedule:
            if input_tokens <= tier.max_input:
                raw = int(input_tokens * tier.ratio)
                return max(tier.min_tokens, min(raw, tier.max_tokens))
        return 2500

    async def evaluate(self, gold_datum, generated_text: str) -> SummaryEval:
        # Phase 1: Operational Guardrails
        if re.search(r"(?i)(this summary|the author|in conclusion)", generated_text):
            return SummaryEval(1.0, True, "Meta-commentary detected")

        if not re.search(r"[.!?]$", generated_text.strip()):
            return SummaryEval(1.0, True, "Truncated output (no terminal punctuation)")

        # Phase 2-4: Concurrent Evaluation
        tasks = [
            self._calc_semantic(generated_text, gold_datum.summary.summary_embedding),
            self._calc_entities(generated_text, gold_datum.summary),
            self._calc_facts(generated_text, gold_datum.summary.key_facts),
            self._calc_structure(generated_text, gold_datum.summary.logical_outline),
            self._calc_length(generated_text, gold_datum.entry.token_count),
        ]

        results = await asyncio.gather(*tasks)

        components = {
            "semantic": results[0],
            "entity": results[1],
            "fact": results[2],
            "structure": results[3],
            "length": results[4],
        }

        total_loss = sum(c.score * c.weight for c in components.values())
        return SummaryEval(
            total_loss=total_loss, is_hard_fail=False, components=components
        )

    async def _calc_semantic(self, text: str, gold_vec: list[float]) -> EvalComponent:
        gen_vec = await self._get_embedding(text)
        dot = numpy.dot(gen_vec, gold_vec)
        norm_gen = numpy.linalg.norm(gen_vec)
        norm_gold = numpy.linalg.norm(gold_vec)
        score = 1.0 - (dot / (norm_gen * norm_gold))
        return EvalComponent(
            score=float(score), weight=0.15, raw_metrics={"cos_dist": score}
        )

    async def _calc_entities(self, text: str, gold) -> EvalComponent:
        entities = await asyncio.to_thread(
            self.gliner.predict_entities, text, gold.entity_list
        )
        if not entities:
            return EvalComponent(1.0, 0.20, {"found": 0})

        tasks = [self._get_embedding(e["text"]) for e in entities]
        gen_vecs = await asyncio.gather(*tasks)

        gold_matrix = numpy.array(gold.entity_list_embeddings)
        gen_matrix = numpy.array(gen_vecs)

        sim_matrix = numpy.dot(gold_matrix, gen_matrix.T) / (
            numpy.linalg.norm(gold_matrix, axis=1)[:, None]
            * numpy.linalg.norm(gen_matrix, axis=1)
        )

        matches = numpy.max(sim_matrix, axis=1) >= self.entity_threshold
        recall = numpy.sum(matches) / len(gold.entity_list)
        return EvalComponent(
            score=float(1.0 - recall), weight=0.20, raw_metrics={"recall": recall}
        )

    async def _calc_facts(self, text: str, gold_facts: list[str]) -> EvalComponent:
        pairs = [[text, fact] for fact in gold_facts]
        results = await asyncio.to_thread(self.nli.predict, pairs)
        entailment_scores = results[:, 2]
        recall = numpy.mean(entailment_scores >= 0.5)
        return EvalComponent(
            score=float(1.0 - recall), weight=0.40, raw_metrics={"fact_recall": recall}
        )

    async def _calc_structure(self, text: str, outline: list[str]) -> EvalComponent:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 2 or len(outline) < 2:
            return EvalComponent(0.0, 0.20, {"inversions": 0})

        pairs = [[sent, point] for point in outline for sent in sentences]
        results = await asyncio.to_thread(self.nli.predict, pairs)
        scores = results[:, 2].reshape(len(outline), len(sentences))
        mapped_indices = numpy.argmax(scores, axis=1)

        inversions = 0
        total_pairs = 0
        for i in range(len(mapped_indices)):
            for j in range(i + 1, len(mapped_indices)):
                total_pairs += 1
                if mapped_indices[i] > mapped_indices[j]:
                    inversions += 1

        norm_inversion_loss = inversions / total_pairs if total_pairs > 0 else 0.0
        return EvalComponent(
            score=float(norm_inversion_loss),
            weight=0.20,
            raw_metrics={"inversions": inversions},
        )

    async def _calc_length(self, text: str, input_tokens: int) -> EvalComponent:
        target = self._get_target_length(input_tokens)
        actual = len(text.split())
        delta = abs(actual - target) / target
        loss = min(1.0, delta**0.5)
        return EvalComponent(
            score=float(loss),
            weight=0.05,
            raw_metrics={"target": target, "actual": actual},
        )


# --- Execution Block ---


async def main():
    from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
    from gliner import GLiNER
    from sentence_transformers import CrossEncoder

    # Initialize dependencies
    hw_client = HeadwaterAsyncClient().embeddings
    gliner_model = GLiNER.from_pretrained("numind/GLiNER_L_v2")
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    schedule = [
        CompressionTier(max_input=2000, ratio=0.15, min_tokens=300, max_tokens=400),
        CompressionTier(max_input=10000, ratio=0.10, min_tokens=400, max_tokens=1000),
        CompressionTier(max_input=40000, ratio=0.05, min_tokens=1000, max_tokens=2000),
    ]

    evaluator = MultiFactorLoss(hw_client, gliner_model, nli_model, schedule)

    # Placeholder: Import your GoldStandardDatum here
    from conduit.strategies.summarize.datasets.load_datasets import load_golden_dataset

    golden_data = load_golden_dataset()
    gold_datum = golden_data[0]

    # gold_datum = ...

    generated_text = "The strategic plan for 2026 focuses on global tech adoption. This initiative is led by the WTO."

    result = await evaluator.evaluate(gold_datum, generated_text)

    if result.is_hard_fail:
        print(f"Hard Fail: {result.fail_reason}")
    else:
        print(f"Total Loss: {result.total_loss}")
        for k, v in result.components.items():
            print(f"{k}: {v.score}")


if __name__ == "__main__":
    asyncio.run(main())
