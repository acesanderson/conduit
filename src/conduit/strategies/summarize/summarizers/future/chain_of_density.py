from conduit.extensions.summarize.strategy import SummarizationStrategy


class RecursiveChainOfDensityStrategy(SummarizationStrategy):
    """
    Single-shot strategy.

    Implements the Iterative Densification workflow.

    Workflow:
    1. Generate an initial verbose summary (Draft 1).
    2. Prompt the model to identify 5-10 specific entities (people, dates, numbers)
       present in the source text but missing from Draft 1.
    3. Prompt the model to fuse these missing entities into Draft 1 without
       increasing the total word count.
    4. Repeat for N recursions (default 3) to maximize information density per token.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...
