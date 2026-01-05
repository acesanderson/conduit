from conduit.extensions.summarize.strategy import SummarizationStrategy


class AtomicPropositionStrategy(SummarizationStrategy):
    """
    One-shot strategy.

    Implements Propositional Decomposition for RAG ingestion.

    Workflow:
    1. Prompt the LLM to decompose the text into independent, self-contained
       atomic statements.
    2. Enforce De-referencing: Replace pronouns ('he', 'it', 'the company') with
       their full named entities ('Guido van Rossum', 'Python 3.11', 'Anthropic').
    3. Return a list of strings suitable for individual vector embedding.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...
