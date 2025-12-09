from pydantic import BaseModel, Field, ConfigDict
from typing import TYPE_CHECKING
from conduit.utils.progress.verbosity import Verbosity
from conduit.config import settings

if TYPE_CHECKING:
    from conduit.storage.cache.protocol import ConduitCache
    from rich.console import Console


class ConduitOptions(BaseModel):
    """
    Muscle Tissue: Runtime configuration for the Conduit application.
    Controls side-effects like logging, caching, and UI, separate from LLM inference parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    verbosity: Verbosity = Field(
        default_factory=lambda: settings.default_verbosity,
        description="Level of output detail (SILENT to DEBUG)",
    )
    cache: "ConduitCache | None" = Field(
        default=None, description="Cache backend for storing/retrieving responses"
    )
    console: "Console | None" = Field(
        default_factory=lambda: settings.default_console,
        description="Rich console instance for TUI output",
    )
