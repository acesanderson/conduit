from typing import TYPE_CHECKING
from dataclasses import dataclass
from conduit.utils.progress.verbosity import Verbosity
from conduit.storage.cache.protocol import ConduitCache
from conduit.storage.repository.protocol import (
    ConversationRepository,
)
from rich.console import Console


@dataclass
class ConduitOptions:
    """
    Muscle Tissue: Runtime configuration for the Conduit application.
    Controls side-effects like logging, caching, and UI, separate from LLM inference parameters.
    """

    project_name: str
    verbosity: Verbosity
    cache: ConduitCache | None
    repository: ConversationRepository | None
    console: Console | None

    # Overrides for request behavior
    use_cache: bool | None = True  # Technically: "if cache exists, use it"
    include_history: bool = True  # Whether to include conversation history
