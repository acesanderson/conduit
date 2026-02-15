"""
# File-Level Docstring for `recursive.py`

Adaptive summarization orchestrator that recursively applies one-shot or map-reduce strategies based on input token count and model context window constraints. The `RecursiveSummarizer` is a `@step`-decorated workflow that intelligently routes text through either direct summarization (when content fits the effective context window via `OneShotSummarizer`) or chunked map-reduce pipelines (when text exceeds the threshold via `MapReduceSummarizer`), then recursively refines intermediate results until the final summary is achieved. All strategy selection and compression parameters are resolved from the Harness configuration layer via `get_param()`, enabling experiment-driven optimization without code changes—users can adjust `model`, `effective_context_window_ratio`, prompt templates, and chunking parameters purely through config dictionaries passed to `ConduitHarness`.

The recursion pattern ensures that large documents are progressively compressed across multiple passes: a map-reduce chunking phase produces intermediate summaries, which are fed back into the orchestrator for another decision cycle, until the output fits the threshold and triggers the one-shot finalizer. Metadata is automatically collected at each step (input token estimates, threshold comparisons, nested step traces) for observability within the Harness.

Usage:
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer

config = {
    "model": "gpt-4o",
    "effective_context_window_ratio": 0.6,
    "OneShotSummarizer.prompt": "Summarize concisely: {{text}}",
    "MapReduceSummarizer.prompt": "Summarize chunk {{chunk_index}}/{{total_chunks}}: {{chunk}}",
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
        # --- 1. RESOLVE EXPERIMENT PARAMS FROM HARNESS ---
        # If these keys exist in the Harness config, they will override the defaults here.
        model = get_param("model", default="gpt3")
        cw_ratio = get_param("effective_context_window_ratio", default=0.5)

        # --- 2. EXECUTE DECISION LOGIC ---
        context_window = self.model_store.get_context_window(model)
        effective_threshold = int(context_window * cw_ratio)

        # Local token count to prevent Ollama's silent truncation
        text_token_size = len(self._tokenizer.encode(text))

        add_metadata("input_tokens_local_est", text_token_size)
        add_metadata("effective_threshold", effective_threshold)

        logger.info(
            f"Recursive check: {text_token_size} tokens vs {effective_threshold} threshold (Model: {model})"
        )

        if text_token_size <= effective_threshold:
            # TRIGGER: One-Shot (The "Finishing" Pass)
            # OneShotSummarizer will pull "OneShotSummarizer.prompt" and "model" from Harness
            logger.info("Executing One-Shot summarization pass.")
            return await OneShotSummarizer()(text=text)

        else:
            # TRIGGER: Map-Reduce (The "Chunking" Pass)
            # MapReduceSummarizer pulls "chunk_size", "overlap", and "MapReduceSummarizer.prompt"
            logger.info("Executing Map-Reduce chunking pass.")
            intermediate_summary = await MapReduceSummarizer()(text=text)

            # --- 3. RECURSE ---
            # By calling self(), we re-enter the @step logic.
            # The Harness trace will show this as a nested step.
            logger.info("Map-Reduce complete. Recursing on intermediate result.")
            return await self(text=intermediate_summary)


if __name__ == "__main__":
    import asyncio
    from conduit.core.workflow.harness import ConduitHarness

    async def main():
        # Instantiate the strategy
        summarizer = RecursiveSummarizer()

        # Create a config with experiment parameters
        config = {
            "model": "gpt3",
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
