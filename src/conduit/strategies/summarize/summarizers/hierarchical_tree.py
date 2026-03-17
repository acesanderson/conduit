"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.hierarchical_tree import HierarchicalTreeSummarizer

config = {
    "model": "gpt-oss:latest",
    "chunk_size": 8000,
    "overlap": 500,
    "group_size": 4,
}

harness = ConduitHarness(config=config)
result = await harness.run(HierarchicalTreeSummarizer(), text=long_document)
```
"""

from __future__ import annotations

import asyncio
import logging
from typing import override, Any
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.chunker import Chunker

logger = logging.getLogger(__name__)


class HierarchicalTreeSummarizer(SummarizationStrategy):
    """
    Bottom-up tree summarization (RAPTOR-lite).

    Workflow:
    1. Chunk the text.
    2. Summarize all chunks in parallel (level 1).
    3. Group those summaries and summarize each group in parallel (level 2).
    4. Repeat until a single root summary remains.

    Every node at a given level is computed concurrently. For very long documents
    where rolling-refine's sequential cost is prohibitive, this is the natural
    upgrade — you get multi-level coherence without serial latency.

    Config params:
        group_size: number of summaries to merge per node at each level (default: 4)
    """

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            group_size = get_param("group_size", default=4)

            # 1. Chunk
            chunker = Chunker()
            chunks = await chunker(text)
            total_chunks = len(chunks)
            logger.info(
                f"HierarchicalTreeSummarizer: {total_chunks} chunks, group_size={group_size}"
            )
            add_metadata("num_chunks", total_chunks)

            if total_chunks == 1:
                logger.info("Single chunk — delegating to OneShotSummarizer")
                return await OneShotSummarizer()(_TextInput(chunks[0]), config)

            # 2. Bottom-up reduction: each level collapses current_level by group_size
            current_level: list[str] = list(chunks)
            level = 0

            while len(current_level) > 1:
                level += 1
                groups = [
                    current_level[i : i + group_size]
                    for i in range(0, len(current_level), group_size)
                ]
                logger.info(
                    f"Level {level}: {len(current_level)} inputs → {len(groups)} groups"
                )
                summaries = await asyncio.gather(
                    *[
                        OneShotSummarizer()(_TextInput("\n\n".join(group)), config)
                        for group in groups
                    ]
                )
                current_level = list(summaries)
                add_metadata(f"level_{level}_nodes", len(groups))

            add_metadata("tree_depth", level)
            return current_level[0]
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)


if __name__ == "__main__":
    import asyncio
    from conduit.core.workflow.harness import ConduitHarness

    async def main():
        summarizer = HierarchicalTreeSummarizer()
        config = {
            "model": "gpt-oss:latest",
            "chunk_size": 4000,
            "overlap": 200,
            "group_size": 3,
        }
        harness = ConduitHarness(config=config)
        test_text = "This is a test sentence with detailed content. " * 300
        result = await harness.run(summarizer, _TextInput(test_text), config)
        print(f"Summary: {result}")
        print(f"Trace: {harness.trace}")

    asyncio.run(main())
