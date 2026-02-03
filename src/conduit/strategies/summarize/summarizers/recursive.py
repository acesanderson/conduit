import logging
import tiktoken
from pathlib import Path
from typing import override

from conduit.utils.logs import logging_config  # noqa: F401

from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.core.workflow.step import step

logger = logging.getLogger(__name__)

default_chunker = Chunker()

EXAMPLE_TEXT = Path(__file__).parent / "example.txt"


class RecursiveSummarizer(SummarizationStrategy):
    """
    Orchestrator strategy that decides between One-Shot and Map-Reduce.
    Uses local token counting to avoid API truncation issues.
    """

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

        # Initialize local tokenizer for heuristic checks
        # We use cl100k_base (GPT-4) as a robust standard reference.
        # Even if the target model is Llama, this gives us a safe, linear order-of-magnitude count.
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @step
    @override
    async def __call__(self, text: str, *args, **kwargs) -> str:
        # Tunable parameters
        effective_context_window = int(
            self.context_window * self.effective_context_window_ratio
        )
        # Note: chunk_size is calculated here but implicitly handled by the MapReduce strategy's own defaults
        # unless we inject it.

        # [CRITICAL FIX] Use local tokenizer.
        # API tokenizers (especially Ollama) truncate inputs > context_window,
        # causing the orchestrator to falsely believe a huge file fits in context.
        text_token_size = len(self._tokenizer.encode(text))

        # [LOGGING] Heuristic
        logger.info(
            f"Input tokens (local est): [bold cyan]{text_token_size}[/bold cyan] | "
            f"Threshold: [bold yellow]{effective_context_window}[/bold yellow] | "
            f"Model: {self.model_name}"
        )

        # Decide strategy based on text token size
        if text_token_size <= effective_context_window:
            # [LOGGING] Decision: One-Shot
            logger.info(
                "Token count within limit. Executing [green]One-Shot[/green] strategy."
            )
            return await self.one_shot(text, self.model_name)
        else:
            # [LOGGING] Decision: Map-Reduce
            logger.info(
                "Token count exceeds limit. Executing [magenta]Map-Reduce[/magenta] strategy."
            )
            return await self.map_reduce(text)

    async def one_shot(self, text: str, model_name: str) -> str:
        from conduit.strategies.summarize.summarizers.one_shot import (
            OneShotSummarizer,
        )

        summarizer = OneShotSummarizer()
        summary = await summarizer(text=text, model=model_name)
        return summary

    async def map_reduce(
        self,
        text: str,
    ) -> str:
        from conduit.strategies.summarize.summarizers.map_reduce import (
            MapReduceSummarizer,
        )

        summarizer = MapReduceSummarizer()
        # MapReduce will handle the chunking internally using its own Chunker
        summary = await summarizer(text=text, model=self.model_name)

        # [LOGGING] Recursion
        logger.info("Map-Reduce pass complete. Recursing on the combined summary.")

        # Recurse: Feed the result back into __call__ to see if it fits yet
        return await self(summary)


if __name__ == "__main__":
    import asyncio
    from conduit.core.workflow.harness import ConduitHarness

    # Sample text for testing
    if EXAMPLE_TEXT.exists():
        sample_text = EXAMPLE_TEXT.read_text()
    else:
        sample_text = "This is a placeholder text for testing." * 500

    async def test_recursive_summarizer():
        # Create the summarizer instance
        # Ensure we use a model name that exists in your setup
        summarizer = RecursiveSummarizer(model_name="gpt-oss:latest")

        # Create a harness with configuration
        config = {
            "model": "gpt-oss:latest",
        }
        harness = ConduitHarness(config=config)

        print("--- Starting Recursive Summarizer ---")
        # Run the summarizer
        result = await harness.run(summarizer, text=sample_text)

        print("\n--- Final Result ---")
        print(result)

        print("\n--- Trace ---")
        harness.view_trace()

        return result

    asyncio.run(test_recursive_summarizer())
