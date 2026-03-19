from __future__ import annotations

import math
import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-8)


class ClusterSelectSummarizer(SummarizationStrategy):
    """
    K-Means cluster-based chunk selection summarizer.

    Workflow:
    1. Chunk the text.
    2. Embed all chunks via HeadwaterAsyncClient.
    3. Run K-Means clustering to find K semantic clusters.
    4. For each cluster, select the chunk closest to the centroid (cosine similarity).
    5. Pass the K selected chunks in original document order to OneShotSummarizer.

    Config params:
        k:               number of clusters (default: min(10, total_chunks))
        embedding_model: model for embeddings (default: all-MiniLM-L6-v2)
        model:           LLM for final summarization (default: gpt3)
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt3"
        k: int | None = None  # None → computed as min(10, total_chunks)
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
        from sklearn.cluster import KMeans
        import numpy as np

        chunker = Chunker()
        chunks = await chunker(text, config)
        total_chunks = len(chunks)
        logger.info(f"ClusterSelectSummarizer: {total_chunks} chunks")

        if total_chunks == 0:
            return ""
        if total_chunks == 1:
            return await OneShotSummarizer()(_TextInput(chunks[0]), config)

        k = cfg.k if cfg.k is not None else min(10, total_chunks)
        k = min(k, total_chunks)

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

        embedding_matrix = np.array(embeddings)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding_matrix)
        centroids = kmeans.cluster_centers_

        # For each cluster, select the chunk closest to the centroid
        selected_indices: list[int] = []
        for cluster_id in range(k):
            member_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            if not member_indices:
                continue
            centroid = centroids[cluster_id].tolist()
            best = max(member_indices, key=lambda i: _cosine_similarity(embeddings[i], centroid))
            selected_indices.append(best)

        selected_indices.sort()  # preserve original document order

        add_metadata("num_chunks", total_chunks)
        add_metadata("k", k)
        add_metadata("chunks_selected", len(selected_indices))

        selected_text = "\n\n".join(chunks[i] for i in selected_indices)
        return await OneShotSummarizer()(_TextInput(selected_text), config)


if __name__ == "__main__":
    import asyncio

    _sample = (
        "The Apollo 11 mission launched on July 16, 1969, carrying astronauts Neil Armstrong, "
        "Buzz Aldrin, and Michael Collins. On July 20, Armstrong and Aldrin landed on the Moon "
        "in the Sea of Tranquility while Collins orbited above. Armstrong became the first human "
        "to walk on the Moon at 02:56 UTC, followed by Aldrin. They collected 21.5 kg of lunar "
        "material and deployed several scientific instruments. The mission returned to Earth on "
        "July 24, splashing down in the Pacific Ocean. It was the fifth crewed mission of NASA's "
        "Apollo program and fulfilled President Kennedy's 1961 goal of landing on the Moon before "
        "the end of the decade."
    )

    async def _main() -> None:
        result = await ClusterSelectSummarizer()(
            _TextInput(_sample), {"model": "gpt3", "k": 3}
        )
        print(result)

    asyncio.run(_main())
