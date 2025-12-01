"""
### Conduit vs. Model: The Division of Labor

The **Model** class is the **Execution Runtime**; it handles the *mechanics* of intelligence—I/O, token accounting, caching, and normalizing disparate API protocols into a unified Request/Response standard. The **Conduit** class is your **Workflow Orchestrator**; it handles the *context* of the application—templating prompts, managing conversation history (`MessageStore`), and governing the specific topology (linear, parallel, or recursive) of the execution flow.

### The Conduit Family Taxonomy

* **`BaseConduit` (Abstract Stem):** The foundational abstract class that defines the core protocol for all conduit topologies, managing prompt rendering, message coercion, and input validation.
* **`SyncConduit` (Linear Blocking):** The standard, synchronous pipeline that binds a prompt to a model for a simple 1-in-1-out execution flow.
* **`AsyncConduit` (Linear Non-Blocking):** Mirrors `SyncConduit` logic but returns awaitable coroutines, allowing it to yield control within an event loop for responsive applications.
* **`BatchConduit` (Parallel):** Manages high-throughput concurrency, mapping a list of inputs to a list of outputs while handling aggregation and partial failures.
* **`ToolConduit` (Cyclic):** Orchestrates a recursive execution loop (Model $\to$ Decision $\to$ Tool $\to$ Result) until a final answer is derived.
* **`SkillsConduit` (Dynamic):** Implements progressive disclosure by analyzing context and mutating the system prompt to inject specific capabilities ("skills") at runtime.
"""

from __future__ import annotations
from conduit.config import settings
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.prompt.prompt import Prompt
    from conduit.model.model_base import ModelBase
    from conduit.parser.parser import Parser
    from conduit.message.messagestore import MessageStore
    from conduit.progress.verbosity import Verbosity

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
            from conduit.message.messagestore import MessageStore

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
