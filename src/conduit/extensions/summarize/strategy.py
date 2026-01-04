from abc import ABC, abstractmethod
from conduit.core.workflow.workflow import step


class SummarizationStrategy(ABC):
    """
    Abstract base class for summarization strategies.
    These are Steps that are ready to roll.
    Each of these need to be wrapped with @step.
    """

    @step
    @abstractmethod
    async def __call__(self, text: str, **kwargs) -> str:
        """
        Execute the summarization workflow.
        """
        ...


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    These are Steps that are ready to roll.
    Each of these need to be wrapped with @step.
    """

    @step
    @abstractmethod
    async def chunk(self, text: str, **kwargs) -> list[str]:
        """
        Execute the chunking workflow.
        Return type should be LLM-ready list of strings (i.e. any necessary metadata should be injected here).
        """
        ...
