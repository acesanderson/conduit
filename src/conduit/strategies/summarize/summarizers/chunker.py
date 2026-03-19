from __future__ import annotations

from typing import override
from pydantic import BaseModel, ConfigDict
from conduit.strategies.summarize.strategy import ChunkingStrategy
from conduit.core.workflow.step import step, add_metadata
import semchunk
import tiktoken
import statistics


class Chunker(ChunkingStrategy):
    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        chunk_size: int = 12000
        overlap: int = 500
        tokenizer_model: str = "gpt-4o"
        memoize: bool = True

    config_model = Config

    @step
    @override
    async def __call__(self, text: str, config: dict) -> list[str]:
        cfg = self.Config(**config)

        try:
            tokenizer = tiktoken.encoding_for_model(cfg.tokenizer_model)
        except KeyError:
            tokenizer = tiktoken.get_encoding("o200k_base")

        def count_tokens(t: str) -> int:
            return len(tokenizer.encode(t))

        chunks = semchunk.chunk(
            text,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
            token_counter=count_tokens,
            memoize=cfg.memoize,
        )

        add_metadata("num_chunks", len(chunks))
        add_metadata("original_text_chars", len(text))

        if chunks:
            avg_chars = statistics.mean(len(c) for c in chunks)
            add_metadata("avg_chunk_chars", int(avg_chars))
            add_metadata("last_chunk_token_est", count_tokens(chunks[-1]))

        return chunks
