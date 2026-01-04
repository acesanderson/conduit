from conduit.extensions.summarize.strategy import SummarizationStrategy


class RollingRefineStrategy(SummarizationStrategy):
    """
    Multi-shot strategy.

    Implements a sequential 'Refine' workflow for narrative continuity.

    Workflow:
    1. Chunk the text linearly.
    2. Summarize Chunk 1.
    3. Loop through remaining chunks:
       Input = (Current Summary) + (New Chunk Text)
       Prompt = "Update the Current Summary with new information from the New Chunk.
                 Do not delete existing relevant info. Do not repeat facts."
    4. Return the final evolved state of the summary.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...
