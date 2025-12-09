from __future__ import annotations
from conduit.config import settings
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.core.prompt.prompt import Prompt
    from conduit.core.model.model_base import ModelBase
    from conduit.core.parser.parser import Parser
    from conduit.domain.message.messagestore import MessageStore
    from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


class ConduitBase:
    def __init__(
        self,
        # Required
        model: ModelBase,
        # Major components
        prompt: Prompt | None = None,
        parser: Parser | None = None,
        message_store: MessageStore | None = None,
        # Project defaults
        console: Console | None = settings.default_console,
        system_message: str | None = settings.system_prompt,
        verbosity: Verbosity = settings.default_verbosity,
    ):
        self.prompt: Prompt | None = prompt
        self.model: ModelBase | None = model
        self.verbosity: Verbosity = verbosity
        self.parser: Parser | None = parser
        self.console: Console | None = console
        self.message_store: MessageStore | None = message_store
        if self.prompt:
            self.input_schema: set[str] = self.prompt.input_schema
        else:
            self.input_schema = set()

    # Config methods (if we want to enable/disable components post-init)
    def enable_message_store(self, name: str) -> None:
        if not self.message_store:
            from conduit.domain.message.messagestore import MessageStore

            self.message_store = MessageStore(name=name)

    def disable_message_store(self) -> None:
        self.message_store = None

    def enable_console(self) -> None:
        if not self.console:
            from rich.console import Console

            self.console = Console()

    def disable_console(self) -> None:
        self.console = None
        self.model.disable_console()

    def enable_cache(self) -> None:
        self.model.enable_cache()

    def disable_cache(self) -> None:
        self.model.disable_cache()
