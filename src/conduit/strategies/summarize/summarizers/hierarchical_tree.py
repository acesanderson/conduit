from __future__ import annotations

import asyncio
import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
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

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        group_size: int = 4
        chunk_size: int = 12000
        overlap: int = 500

    config_model = Config

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data

        chunker = Chunker()
        chunks = await chunker(text, config)
        total_chunks = len(chunks)
        logger.info(
            f"HierarchicalTreeSummarizer: {total_chunks} chunks, group_size={cfg.group_size}"
        )
        add_metadata("num_chunks", total_chunks)

        if total_chunks == 1:
            logger.info("Single chunk — delegating to OneShotSummarizer")
            return await OneShotSummarizer()(_TextInput(chunks[0]), config)

        current_level: list[str] = list(chunks)
        level = 0

        while len(current_level) > 1:
            level += 1
            groups = [
                current_level[i : i + cfg.group_size]
                for i in range(0, len(current_level), cfg.group_size)
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
