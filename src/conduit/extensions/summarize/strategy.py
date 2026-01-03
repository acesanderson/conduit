from abc import ABC, abstractmethod
from typing import Any


class SummarizationStrategy(ABC):
    """
    Abstract base class for all summarization logic in Siphon.
    """

    @abstractmethod
    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str:
        """
        Execute the summarization workflow.
        """
        ...


class PolishingStrategy(ABC):
    """
    Abstract base class for formatting/styling the final output.
    """

    @abstractmethod
    def polish(self, raw_summary: str, metadata: dict[str, Any]) -> str:
        """
        Apply specific formatting templates (Markdown, JSON, etc.) to the summary.
        """
        ...


# ==============================================================================
# Tier 1: The "Single-Shot" Zone (< Context Window)
# Goal: Maximum Semantic Density
# ==============================================================================


class RecursiveChainOfDensityStrategy(SummarizationStrategy):
    """
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


class SchemaExtractionStrategy(SummarizationStrategy):
    """
    Implements structured data extraction using strictly enforced schemas.

    Workflow:
    1. Define a Pydantic model or JSON schema representing the desired output
       (e.g., list[ActionItem], list[Argument], list[Decision]).
    2. Force the LLM (via grammar constraints or function calling) to output
       only valid JSON adhering to that schema.
    3. Parse and validate the output.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...


class AtomicPropositionStrategy(SummarizationStrategy):
    """
    Implements Propositional Decomposition for RAG ingestion.

    Workflow:
    1. Prompt the LLM to decompose the text into independent, self-contained
       atomic statements.
    2. Enforce De-referencing: Replace pronouns ('he', 'it', 'the company') with
       their full named entities ('Guido van Rossum', 'Python 3.11', 'Anthropic').
    3. Return a list of strings suitable for individual vector embedding.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...


# ==============================================================================
# Tier 2: The "Overflow" Zone (> Context Window, Linear)
# Goal: Coherence & Narrative Preservation
# ==============================================================================


class RollingRefineStrategy(SummarizationStrategy):
    """
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


class MapDedupeReduceStrategy(SummarizationStrategy):
    """
    Implements a parallel 'Map-Reduce' workflow optimized for extraction.

    Workflow:
    1. Chunk the text.
    2. Map (Parallel): Run a distinct extraction prompt (e.g., "list all tasks")
       on every chunk simultaneously using a lightweight model.
    3. Reduce: Collect all extraction lists.
    4. Dedupe: Run a final LLM pass to merge duplicates and normalize format.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...


# ==============================================================================
# Tier 3: The "Massive" Zone (Books, Multi-hour Video)
# Goal: Navigability & Gist
# ==============================================================================


class ClusterSelectStrategy(SummarizationStrategy):
    """
    Implements Embedding-based Clustering for 'Snapshot' generation.

    Workflow:
    1. Chunk the entire document.
    2. Generate vector embeddings for all chunks (using a fast local model).
    3. Run K-Means clustering to identify K distinct semantic topics (e.g., K=10).
    4. Select the 'Centroid' chunk from each cluster (the most representative text).
    5. Feed the list of Centroids to the LLM to generate a 'Table of Contents'
       or high-level overview.
    """

    def summarize(self, text: str, context_limit: int = 8192, **kwargs) -> str: ...


class HierarchicalTreeStrategy(SummarizationStrategy):
    """
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


# ==============================================================================
# The Finishing Layers
# ==============================================================================


class YouTubeCompanionStrategy(PolishingStrategy):
    """
    Formats the output for video consumption.

    Structure:
    - Hook / Teaser Paragraph
    - Timestamped Chapter Markers (derived from source metadata if available)
    - Key Takeaways (Bulleted)
    - Further Reading / Referenced URLs
    """

    def polish(self, raw_summary: str, metadata: dict[str, Any]) -> str: ...


class ObsidianVaultStrategy(PolishingStrategy):
    """
    Formats the output for Personal Knowledge Management (PKM).

    Structure:
    - Frontmatter (YAML tags, source link, date)
    - [[Wikilinks]] for key entities found in the text.
    - Markdown Headers (#, ##) for hierarchy.
    - 'Related Concepts' section at the bottom.
    """

    def polish(self, raw_summary: str, metadata: dict[str, Any]) -> str: ...


class ExecutiveBriefStrategy(PolishingStrategy):
    """
    Formats the output for decision making (BLUF).

    Structure:
    - BLUF (Bottom Line Up Front): 1 sentence thesis.
    - Critical Implications: 3 bullet points.
    - Decision Vectors / Action Items.
    """

    def polish(self, raw_summary: str, metadata: dict[str, Any]) -> str: ...
