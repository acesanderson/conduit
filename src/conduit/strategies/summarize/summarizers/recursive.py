from __future__ import annotations

import logging
import tiktoken
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.map_reduce import MapReduceSummarizer
from conduit.core.model.models.modelstore import ModelStore

logger = logging.getLogger(__name__)


class RecursiveSummarizer(SummarizationStrategy):
    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        effective_context_window_ratio: float = 0.8
        chunk_size: int = 12000
        overlap: int = 500

    config_model = Config

    def __init__(self):
        self.model_store = ModelStore()
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data

        allocated_window = ModelStore.get_num_ctx(cfg.model)
        effective_threshold = int(allocated_window * cfg.effective_context_window_ratio)
        text_token_size = len(self._tokenizer.encode(text))

        add_metadata("num_ctx_allocated", allocated_window)
        add_metadata("effective_chunk_size", effective_threshold)
        add_metadata("current_input_tokens", text_token_size)

        logger.info(
            f"Recursive Summarizer Check: {text_token_size} tokens vs {effective_threshold} threshold."
        )

        if text_token_size <= effective_threshold:
            logger.info(
                f"Input ({text_token_size}) fits in threshold ({effective_threshold}). Running One-Shot."
            )
            return await OneShotSummarizer()(_TextInput(text), config)
        else:
            logger.info(
                f"Input ({text_token_size}) exceeds threshold. Running Map-Reduce."
            )
            intermediate_summary = await MapReduceSummarizer()(
                _TextInput(text),
                {**config, "chunk_size": effective_threshold},
            )
            logger.info("Intermediate summary complete. Recursing to check size.")
            return await self(_TextInput(intermediate_summary), config)
