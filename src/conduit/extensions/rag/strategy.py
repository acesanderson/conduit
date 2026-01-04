"""
Here we define an ABC for RAG (Retrieval-Augmented Generation) strategies.
"""

from abc import ABC, abstractmethod
from conduit.core.workflow.workflow import step


class RAGStrategy(ABC):
    """
    Abstract base class for RAG (Retrieval-Augmented Generation) strategies.
    These are Steps that are ready to roll.
    Each of these need to be wrapped with @step.
    """

    @step
    @abstractmethod
    async def __call__(self, query: str, **kwargs) -> str:
        """
        Execute the RAG workflow.
        """
        ...
