from conduit.extensions.summarize.strategy import SummarizationStrategy


class SchemaExtractionStrategy(SummarizationStrategy):
    """
    One-shot strategy.

    Implements structured data extraction using strictly enforced schemas.

    Workflow:
    1. Define a Pydantic model or JSON schema representing the desired output
       (e.g., list[ActionItem], list[Argument], list[Decision]).
    2. Force the LLM (via grammar constraints or function calling) to output
       only valid JSON adhering to that schema.
    3. Parse and validate the output.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...
