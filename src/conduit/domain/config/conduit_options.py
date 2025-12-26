from conduit.utils.progress.verbosity import Verbosity
from conduit.storage.cache.protocol import ConduitCache
from conduit.storage.repository.protocol import (
    ConversationRepository,
)
from conduit.storage.repository.persistence_mode import PersistenceMode
from pydantic import BaseModel, Field, ConfigDict, field_validator
from rich.console import Console


class ConduitOptions(BaseModel):
    """
    Muscle Tissue: Runtime configuration for the Conduit application.
    Controls side-effects like logging, caching, and UI, separate from LLM inference parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_name: str
    verbosity: Verbosity = Field(
        default=Verbosity.PROGRESS, description="Verbosity level for logging and UI."
    )
    cache: ConduitCache | None = Field(
        default=None,
        description="Cache backend for storing/retrieving generations.",
        exclude=True,
    )
    repository: ConversationRepository | None = Field(
        default=None,
        description="Repository backend for persisting conversations.",
        exclude=True,
    )
    console: Console | None = Field(
        default=None,
        description="Rich console for enhanced logging/UI.",
        exclude=True,
    )

    # Overrides for request behavior
    use_cache: bool | None = True  # Technically: "if cache exists, use it"
    include_history: bool = True  # Whether to include conversation history
    persistence_mode: PersistenceMode = PersistenceMode.RESUME

    # Dev options
    debug_payload: bool = False  # Log full request/response payloads for debugging

    @field_validator("verbosity")
    @classmethod
    def verbosity_must_not_be_none(cls, v):
        if v is None:
            raise ValueError("verbosity cannot be None")
        return v
