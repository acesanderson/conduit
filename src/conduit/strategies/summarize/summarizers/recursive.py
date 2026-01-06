from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.core.workflow.workflow import step, get_param, add_metadata
from typing import override

default_chunker = Chunker()


class RecursiveSummarizer(SummarizationStrategy):
    def __init__(
        self,
        model_name: str = "gpt3",
        chunker: Chunker = default_chunker,
        effective_context_window_ratio: float = 0.5,
        chunk_size_ratio: float = 0.5,
    ):
        from conduit.core.model.models.modelstore import ModelStore

        self.model_store: ModelStore = ModelStore()
        self.model_name: str = self.model_store.validate_model(model_name)
        self.chunker: Chunker = chunker
        self.effective_context_window_ratio: float = effective_context_window_ratio
        self.chunk_size_ratio: float = chunk_size_ratio
        self.context_window: int = self.model_store.get_context_window(model_name)

    @step
    @override
    async def __call__(self, text: str, *args, **kwargs) -> str:
        from conduit.core.model.model_async import ModelAsync

        # Tunable parameters
        effective_context_window = int(
            self.context_window * self.effective_context_window_ratio
        )
        chunk_size = int(effective_context_window * self.chunk_size_ratio)

        # Get text token size
        model = ModelAsync(model_name)
        text_token_size = await model.tokenize(text)

        # Decide strategy based on text token size
        if text_token_size <= effective_context_window:
            # Use one-shot summarization and terminate
            return await self.one_shot(text, model_name)
        else:
            # Chunk text and use map-reduce summarization; recurse
            return await self.map_reduce(text)

    async def chunk_text(self, text: str, chunk_size: int):
        chunks = await self.chunker(text=text, chunk_size=chunk_size)
        return chunks

    async def one_shot(self, text: str, model_name: str) -> str:
        from conduit.strategies.summarize.summarizers.one_shot import (
            OneShotSummarizer,
        )

        summarizer = OneShotSummarizer()
        summary = await summarizer(text=text, model_name=model_name)
        return summary

    async def map_reduce(
        self,
        text: str,
    ) -> str:
        from conduit.strategies.summarize.summarizers.map_reduce import (
            MapReduceSummarizer,
        )

        summarizer = MapReduceSummarizer()
        summary = await summarizer(text=text, model_name=self.model_name)
        return await self.process(summary)  # Recurse
