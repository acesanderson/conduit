"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer

config = {
    "model": "gpt-oss:latest",
    "effective_context_window_ratio": 0.6,
    "OneShotSummarizer.prompt": "Summarize concisely: {{ text }}",
    "MapReduceSummarizer.prompt": "Summarize chunk {{ chunk_index }}/{{ total_chunks }}: {{ chunk }}",
    "chunk_size": 8000,
    "overlap": 500,
}

harness = ConduitHarness(config=config)
result = await harness.run(RecursiveSummarizer(), text=long_document)
```
"""

import logging
import tiktoken
from typing import override
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.map_reduce import MapReduceSummarizer
from conduit.core.model.models.modelstore import ModelStore

logger = logging.getLogger(__name__)


class RecursiveSummarizer(SummarizationStrategy):
    """
    A recursive orchestrator that pulls all configuration from the Harness.
    """

    def __init__(self):
        # We only keep things that are technically static across all experiments
        self.model_store = ModelStore()
        # Using a standard reference encoding for local heuristic checks
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @step
    @override
    async def __call__(self, text: str, *args, **kwargs) -> str:
        # 1. Resolve Params from Harness
        cw_ratio = get_param("effective_context_window_ratio", default=0.8)
        model_slug = get_param("model", default="gpt-oss:latest")

        # 2. Get the ACTUAL allocated context window (Truth is Local)
        # This calls your ModelStore.get_num_ctx which now reads ollama_context_sizes.json
        allocated_window = ModelStore.get_num_ctx(model_slug)

        # 3. Calculate your chunking threshold
        # Example: 65,536 * 0.8 = 52,428 tokens
        effective_threshold = int(allocated_window * cw_ratio)

        # 4. Tokenize local input to decide: Summarize or Chunk?
        text_token_size = len(self._tokenizer.encode(text))

        add_metadata("num_ctx_allocated", allocated_window)
        add_metadata("effective_chunk_size", effective_threshold)
        add_metadata("current_input_tokens", text_token_size)

        logger.info(
            f"Recursive Summarizer Check: {text_token_size} tokens vs {effective_threshold} threshold."
        )

        # --- THE DECISION ENGINE ---

        if text_token_size <= effective_threshold:
            # BASE CASE: Text fits. Perform final high-fidelity summary.
            logger.info(
                f"Input ({text_token_size}) fits in threshold ({effective_threshold}). Running One-Shot."
            )
            return await OneShotSummarizer()(text=text)

        else:
            # RECURSIVE STEP: Text is too big.
            # 1. Map-Reduce it into a smaller intermediate summary.
            logger.info(
                f"Input ({text_token_size}) exceeds threshold. Running Map-Reduce."
            )
            intermediate_summary = await MapReduceSummarizer()(
                text=text,
                chunk_size=effective_threshold,  # Ensure chunker respects our 5090 settings
            )

            # 2. RECURSE: Feed the intermediate summary back into THIS function.
            # This will repeat until the summary is small enough for One-Shot.
            logger.info("Intermediate summary complete. Recursing to check size.")
            return await self(text=intermediate_summary)


if __name__ == "__main__":
    import asyncio
    from conduit.core.workflow.harness import ConduitHarness

    async def main():
        # Instantiate the strategy
        summarizer = RecursiveSummarizer()

        # Create a config with experiment parameters
        config = {
            "model": "gpt-oss:latest",
            "effective_context_window_ratio": 0.5,
        }

        # Create harness and run
        harness = ConduitHarness(config=config)

        # Test with a simple text input
        test_text = "This is a test document. " * 100
        result = await harness.run(summarizer, text=test_text)

        print(f"Summary: {result}")
        print(f"Trace: {harness.trace}")

    asyncio.run(main())
