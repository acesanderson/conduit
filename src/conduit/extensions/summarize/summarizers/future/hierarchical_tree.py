from conduit.extensions.summarize.strategy import SummarizationStrategy


class HierarchicalTreeStrategy(SummarizationStrategy):
    """
    Massive multi-shot strategy.

    Implements a RAPTOR-lite recursive summarization tree.

    Workflow:
    1. Bottom-Up: Chunk the text.
    2. Level 1: Summarize every chunk (optionally using Chain of Density).
    3. Grouping: Concatenate Level 1 summaries into groups that fit context.
    4. Level 2: Summarize the groups.
    5. Repeat until the entire corpus is compressed into a single Root summary.
    6. Return the Root summary (or the traversed tree for deep context).
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...
