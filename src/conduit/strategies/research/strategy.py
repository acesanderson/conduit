"""
Here we define an ABC for research strategies.
Examples:
- Deep Research (wraps LLM providers like Google)
- Web Research
- Literature Review Strategy
- RSS Feed Analysis Strategy
"""

from abc import ABC, abstractmethod
from conduit.core.workflow.workflow import step


class ResearchStrategy(ABC):
    """
    Abstract base class for research strategies.
    These are Steps that are ready to roll.
    Each of these need to be wrapped with @step.
    """

    @step
    @abstractmethod
    async def __call__(self, topic: str, **kwargs) -> str:
        """
        Execute the research workflow.
        """
        ...
