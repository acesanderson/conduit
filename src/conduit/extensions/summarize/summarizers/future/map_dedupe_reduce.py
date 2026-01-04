from conduit.extensions.summarize.strategy import SummarizationStrategy


class MapDedupeReduceStrategy(SummarizationStrategy):
    """
    Multi-shot strategy.

    Implements a parallel 'Map-Reduce' workflow optimized for extraction.

    Workflow:
    1. Chunk the text.
    2. Map (Parallel): Run a distinct extraction prompt (e.g., "list all tasks")
       on every chunk simultaneously using a lightweight model.
    3. Reduce: Collect all extraction lists.
    4. Dedupe: Run a final LLM pass to merge duplicates and normalize format.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...
