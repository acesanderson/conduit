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
        map_model: str | None = None       # chunk summarization model; None = same as model
        map_host_alias: str | None = None  # chunk summarization host; None = same as host_alias
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
            map_model = cfg.map_model or cfg.model
            map_allocated_window = ModelStore.get_num_ctx(map_model)
            map_chunk_size = int(map_allocated_window * cfg.effective_context_window_ratio)
            map_config = {**config, "model": map_model, "chunk_size": map_chunk_size}
            if cfg.map_host_alias is not None:
                map_config["host_alias"] = cfg.map_host_alias
            intermediate_summary = await MapReduceSummarizer()(
                _TextInput(text), map_config,
            )
            logger.info("Intermediate summary complete. Recursing to check size.")
            return await self(_TextInput(intermediate_summary), config)
