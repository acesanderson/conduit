from __future__ import annotations

import math
import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.chunker import Chunker

logger = logging.getLogger(__name__)


def _centroid(embeddings: list[list[float]]) -> list[float]:
    n = len(embeddings)
    dim = len(embeddings[0])
    return [sum(embeddings[i][d] for i in range(n)) / n for d in range(dim)]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-8)


class ExtractivePreFilterSummarizer(SummarizationStrategy):
    """
    Embedding-based extractive pre-filter before LLM summarization.

    Workflow:
    1. Chunk the text.
    2. Embed all chunks via HeadwaterClient.
    3. Compute the document centroid (mean of all chunk embeddings).
    4. Score each chunk by cosine similarity to the centroid.
    5. Retain the top `keep_ratio` fraction in original document order.
    6. Pass the filtered text to OneShotSummarizer.

    Effective for information-sparse documents (transcripts with filler,
    padded reports, legal boilerplate) where large portions of the text
    carry little semantic weight. LLM cost does not scale linearly with
    document length.

    Config params:
        keep_ratio:      fraction of chunks to retain (default: 0.3)
        embedding_model: Headwater embedding model (default: all-MiniLM-L6-v2)
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        keep_ratio: float = 0.3
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        chunk_size: int = 12000
        overlap: int = 500

    config_model = Config

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data

        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        from headwater_api.classes import EmbeddingsRequest, ChromaBatch

        chunker = Chunker()
        chunks = await chunker(text, config)
        total_chunks = len(chunks)
        logger.info(
            f"ExtractivePreFilterSummarizer: {total_chunks} chunks, keep_ratio={cfg.keep_ratio}"
        )
        add_metadata("num_chunks_before_filter", total_chunks)

        if total_chunks == 1:
            logger.info("Single chunk — skipping filter, delegating to OneShotSummarizer")
            return await OneShotSummarizer()(_TextInput(chunks[0]), config)

        request = EmbeddingsRequest(
            model=cfg.embedding_model,
            batch=ChromaBatch(
                ids=[str(i) for i in range(total_chunks)],
                documents=chunks,
            ),
        )
        async with HeadwaterAsyncClient() as client:
            response = await client.embeddings.generate_embeddings(request)
        embeddings = response.embeddings

        centroid = _centroid(embeddings)
        scores = [_cosine_similarity(emb, centroid) for emb in embeddings]

        keep_n = max(1, int(total_chunks * cfg.keep_ratio))
        ranked = sorted(range(total_chunks), key=lambda i: scores[i], reverse=True)
        selected_indices = sorted(ranked[:keep_n])

        logger.info(f"Kept {keep_n}/{total_chunks} chunks after extractive filter")
        add_metadata("num_chunks_after_filter", keep_n)
        add_metadata("filter_reduction_ratio", round(1.0 - keep_n / total_chunks, 3))

        filtered_text = "\n\n".join(chunks[i] for i in selected_indices)
        return await OneShotSummarizer()(_TextInput(filtered_text), config)
