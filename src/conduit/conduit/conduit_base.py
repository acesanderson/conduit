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
    from conduit.result.response import Response
    from conduit.result.error import ConduitError
    from conduit.result.result import ConduitResult
    from conduit.parser.parser import Parser
    from conduit.message.messages import Messages
    from conduit.message.message import Message
    from conduit.message.textmessage import TextMessage
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

    # Validation methods
    def _validate_input_variables(self, input_variables: dict[str, str]) -> None:
        """
        Validates that the provided input variables match the expected input schema.

        Args:
            input_variables (dict): A dictionary of input variables to validate.

        Raises:
            ValueError: If any required input variables are missing.
            ValueError: If any extra input variables are provided that are not
                expected by the prompt.
        """
        # Determine if prompt is expecting variables that are not provided
        missing_vars: set[str] = self.input_schema - input_variables.keys()
        if missing_vars:
            raise ValueError(
                f'Prompt is missing required input variable(s): "{'", "'.join(missing_vars)}"'
            )
        # Determine if extra variables are provided that the prompt does not expect
        extra_vars: set[str] = input_variables.keys() - self.input_schema
        if extra_vars:
            raise ValueError(
                f'Provided input variable(s) are not referenced in prompt: "{'", "'.join(extra_vars)}"'
            )
        return

    def run(
        self,
        # Inputs
        input_variables: dict[str, str] | None = None,
        messages: Messages | list[Message] | None = None,
        parser: Parser | None = None,
        # Configs
        verbose: Verbosity = Verbosity.PROGRESS,
        stream: bool = False,
        cache: bool = True,
        index: int = 0,
        total: int = 0,
        # History / messagestore-related
        include_history: bool = True,
        save: bool = True,
    ) -> ConduitResult:
        # Render our prompt with the input_variables if variables are passed.
        if input_variables and self.prompt:
            self._validate_input_variables(input_variables)
            logger.info("Rendering prompt with input variables: %s", input_variables)
            prompt = self.prompt.render(input_variables=input_variables)
        elif self.prompt:
            logger.info("Using prompt without input variables.")
            prompt = self.prompt.prompt_string
        else:
            logger.info("No prompt provided, using None.")
            prompt = None
        # Resolve messages: messagestore or user-provided messages. Are we including history?
        if SyncConduit.message_store and include_history:
            if messages is not None:
                raise ValueError(
                    "Both messages and message store are provided, please use only one."
                )
            logger.info("Using message store for messages.")
            messages = SyncConduit.message_store.messages
        else:
            if messages is None:
                logger.info("No message store or messages provided, starting fresh.")
                messages = []
            else:
                logger.info("Using user-provided messages.")
        # Coerce messages and query_input into a list of Message objects
        logger.info("Coercing messages and prompt into a list of Message objects.")
        messages = self._coerce_messages_and_prompt(prompt=prompt, messages=messages)
        assert len(messages) > 0, "No messages provided, cannot run conduit."
        # Route input; if string, if message
        logger.info("Querying model.")
        # Save this for later
        result = self.model.query(
            query_input=messages,
            response_model=self.parser.pydantic_model if self.parser else None,
            verbose=verbose,
            cache=cache,
            index=index,
            total=total,
            stream=stream,
        )
        logger.info(f"ModelSync query completed, return type: {type(result)}.")
        # Save to messagestore if we have one and if we have a response.
        if SyncConduit.message_store and save:
            if isinstance(result, Response):
                logger.info("Saving response to message store.")
                if not include_history:
                    # For a one-off query, the user message that initiated it was not
                    # in the message store. We must add it now to maintain history integrity.
                    user_message = messages[
                        -1
                    ]  # The user message is the last in the list sent to the model
                    SyncConduit.message_store.append(user_message)
                SyncConduit.message_store.append(result.message)
                # Context window information
                SyncConduit.message_store.input_tokens += result.input_tokens
                SyncConduit.message_store.output_tokens += result.output_tokens
                SyncConduit.message_store.last_used_model_name = self.model.model
            elif isinstance(result, ConduitError):
                logger.error("ConduitError encountered, not saving to message store.")
                SyncConduit.message_store.query_failed()
        else:
            if not SyncConduit.message_store:
                logger.info("No message store associated with conduit, skipping save.")
            if not save:
                logger.info("'save' is False, skipping save to message store.")
        if not isinstance(result, ConduitResult):
            logger.warning(
                "Result is not a Response or ConduitError: type {type(result)}."
            )
        return result

    def _coerce_messages_and_prompt(
        self, prompt: str | Message | None, messages: Messages | list[Message] | None
    ) -> list[Message] | Messages:
        """
        We want a list of messages to submit to ModelSync.query.
        If we have a prompt, we want to convert it into a user message and append it to messages.
        WARNING: This function will mutate its inputs. If messages = a MessageStore, you might see doubled messages if you don't take this into account. We are taking advantage of mutability and inheritance here but beware -- especially if MessageStore is persistent, as this can also kick off database writes.
        """
        if not messages:
            messages = []
        if isinstance(prompt, Message):
            # If we have a message, just append it
            messages.append(prompt)
        elif isinstance(prompt, str):
            # If we have a string, convert it to a TextMessage
            messages.append(TextMessage(role="user", content=prompt))
        elif prompt is None:
            # If we have no prompt, do nothing
            pass
        else:
            raise ValueError(f"Unsupported query_input type: {type(query_input)}")

        return messages
