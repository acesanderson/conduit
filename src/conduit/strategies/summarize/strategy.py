from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from conduit.core.workflow.protocols import Strategy


@dataclass
class _TextInput:
    """Lightweight input wrapper for internal sub-calls between summarizers."""
    data: str
    source_id: str = "internal"


class SummarizationStrategy(Strategy, ABC):
    """
    Abstract base class for summarization strategies.
    These are Steps that are ready to roll.
    Each of these need to be wrapped with @step.
    """

    @abstractmethod
    async def __call__(self, input: Any, config: dict) -> str:
        """
        Execute the summarization workflow.
        input.data contains the text to summarize.
        config contains strategy parameters (model, prompt, etc.).
        """
        ...


class ChunkingStrategy(Strategy, ABC):
    """
    Abstract base class for chunking strategies.
    These are Steps that are ready to roll.
    Each of these need to be wrapped with @step.
    """

    @abstractmethod
    async def __call__(self, text: str, **kwargs) -> list[str]:
        """
        Execute the chunking workflow.
        Return type should be LLM-ready list of strings (i.e. any necessary metadata should be injected here).
        """
        ...
