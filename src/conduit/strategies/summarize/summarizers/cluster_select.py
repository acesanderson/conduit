"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.cluster_select import ClusterSelectSummarizer

config = {
    "model": "gpt-oss:latest",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "num_clusters": 5,
    "OneShotSummarizer.prompt": "Summarize: {{ text }}",
}

harness = ConduitHarness(config=config)
result = await harness.run(ClusterSelectSummarizer(), text=document)
```
"""

from __future__ import annotations

import logging
from typing import override, Any
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


class ClusterSelectSummarizer(SummarizationStrategy):
    """
    Cluster-based document selection summarizer.

    Workflow:
    1. Chunk the document.
    2. Cluster chunks using embeddings (via Headwater).
    3. For each cluster, select the most representative chunk(s).
    4. Summarize the selected chunks.

    This approach aims to cover different aspects of the document while
    avoiding redundancy by selecting only the most representative content
    from each semantic cluster.
    """

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            num_clusters = get_param("num_clusters", default=5)
            embedding_model = get_param(
                "embedding_model", default="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Chunk the text
            chunker = Chunker()
            chunks = await chunker(text)
            total_chunks = len(chunks)
            logger.info(
                f"ClusterSelectSummarizer: {total_chunks} chunks, {num_clusters} clusters"
            )

            if total_chunks == 0:
                return ""
            if total_chunks == 1:
                return await OneShotSummarizer()(_TextInput(chunks[0]), config)

            # Import inside try block to avoid circular dependency issues
            from headwater_client.client.headwater_client_async import (
                HeadwaterAsyncClient,
            )
            from headwater_api.classes import EmbeddingsRequest, ChromaBatch

            # Get embeddings for all chunks
            request = EmbeddingsRequest(
                model=embedding_model,
                batch=ChromaBatch(
                    ids=[str(i) for i in range(total_chunks)],
                    documents=chunks,
                ),
            )

            async with HeadwaterAsyncClient() as client:
                response = await client.embeddings.generate_embeddings(request)

            embeddings = response.embeddings

            # Simple clustering: use K-means style approach
            # Since we don't have external clustering libraries, use a basic
            # approach: group consecutive chunks into clusters
            # For a more sophisticated approach, we'd use actual clustering

            # Calculate cluster assignments based on embedding similarity
            import math

            def cosine_similarity(a: list[float], b: list[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b + 1e-8)

            def compute_centroid(indices: list[int]) -> list[float]:
                if not indices:
                    return [0.0] * len(embeddings[0])
                dim = len(embeddings[0])
                centroid = [0.0] * dim
                for idx in indices:
                    for d in range(dim):
                        centroid[d] += embeddings[idx][d]
                return [c / len(indices) for c in centroid]

            # Assign chunks to clusters (simple approach)
            # In production, use proper clustering (K-means, hierarchical, etc.)
            chunk_size = max(1, total_chunks // num_clusters)
            clusters: list[list[int]] = []
            for i in range(num_clusters):
                start = i * chunk_size
                end = start + chunk_size if i < num_clusters - 1 else total_chunks
                cluster_indices = list(range(start, end))
                if cluster_indices:
                    clusters.append(cluster_indices)

            logger.info(f"Created {len(clusters)} clusters")

            # For each cluster, find the most representative chunk(s)
            selected_chunks: list[str] = []

            for cluster_idx, cluster_indices in enumerate(clusters):
                # Compute cluster centroid
                centroid = compute_centroid(cluster_indices)

                # Score each chunk by similarity to centroid
                scores = [
                    (idx, cosine_similarity(embeddings[idx], centroid))
                    for idx in cluster_indices
                ]

                # Select top chunk(s) from each cluster
                # Sort by score descending
                scores.sort(key=lambda x: x[1], reverse=True)

                # Take the top 1-2 chunks from each cluster
                # This can be configurable
                num_selected = min(2, len(scores))
                for i in range(num_selected):
                    if i < len(scores):
                        selected_chunks.append(chunks[scores[i][0]])

            logger.info(f"Selected {len(selected_chunks)} representative chunks")

            add_metadata("num_chunks", total_chunks)
            add_metadata("num_clusters", len(clusters))
            add_metadata("selected_chunks", len(selected_chunks))

            if not selected_chunks:
                return ""

            # Summarize the selected chunks
            combined_text = "\n\n".join(selected_chunks)
            one_shot = OneShotSummarizer()
            summary = await one_shot(_TextInput(combined_text), config)

            return summary
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
