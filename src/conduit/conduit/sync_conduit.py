"""
A Conduit is a convenience wrapper for models, prompts, parsers, messages, and response objects.
A conduit needs to have at least a prompt and a model.
Conduits are immutable, treat them like tuples.
"""

# The rest of our package.
from conduit.prompt.prompt import Prompt
from conduit.model.model import Model
from conduit.result.response import Response
from conduit.result.error import ConduitError
from conduit.result.result import ConduitResult
from conduit.parser.parser import Parser
from conduit.message.message import Message
from conduit.message.textmessage import TextMessage
from conduit.message.messages import Messages
from conduit.message.messagestore import MessageStore
from conduit.progress.verbosity import Verbosity

from conduit.logs.logging_config import configure_logging
from typing import TYPE_CHECKING, Optional
import logging

# Our TYPE_CHECKING imports, these ONLY load for IDEs, so you can still lazy load in production.
if TYPE_CHECKING:
    from rich.console import Console

logger = configure_logging(
    # level=logging.INFO,
    # level=logging.NOTSET,
    # level=logging.DEBUG,
    level=logging.CRITICAL,
)


class SyncConduit:
    """
    How we chain things together.
    Instantiate with:
    - a prompt (a string that is ready for jinja2 formatting),
    - a model (a name of a model (full list of accessible models in Model.models))
    - a parser (a function that takes a string and returns a string)
    """

    # If you want logging, initialize a message store with log_file_path parameter, and assign it to your Conduit class as a singleton.
    _message_store: Optional[MessageStore] = None
    # If you want rich progress reporting, add a rich.console.Console object to Conduit. (also can be added at Model level)
    _console: Optional["Console"] = None

    def __init__(
        self, model: Model, prompt: Prompt | None = None, parser: Parser | None = None
    ):
        self.prompt = prompt
        self.model = model
        self.parser = parser
        if self.prompt:
            self.input_schema = self.prompt.input_schema  # this is a set
        else:
            self.input_schema = set()

    def run(
        self,
        input_variables: dict | None = None,
        messages: Messages | list[Message] | None = None,
        parser: Parser | None = None,
        verbose: Verbosity = Verbosity.PROGRESS,
        stream: bool = False,
        cache: bool = True,
        index: int = 0,
        total: int = 0,
    ) -> ConduitResult:
        """
        Executes the Conduit, processing the prompt and interacting with the language model.

        whether 'messages' are provided or if a streaming response is requested.
        It renders the prompt with `input_variables` if a `Prompt` object is
        associated with the Conduit.

        Args:
            input_variables (dict | None): A dictionary of variables to render
                the prompt template. Required if the Conduit's prompt contains
                Jinja2 placeholders. Defaults to None.
            messages (list[Message] | None): A list of `Message` objects
                representing a conversation history or a single message. If
                provided, the Conduit will operate in chat mode. Defaults to an
                empty list.
            parser: (Parser | None): A parser to process the model's output.
            verbose (bool): If True, displays progress information during the
                model query. This is managed by the `progress_display` decorator
                on the underlying `Model.query` call. Defaults to True.
            stream (bool): If True, attempts to stream the response from the
                model. Note that streaming requests do not return a `Response`
                object directly but rather a generator. Defaults to False.
            cache (bool): If True, the response will be cached if caching is
                enabled on the `Model` class. Defaults to True.
            index (int): The current index of the item being processed in a
                batch operation. Used for progress display (e.g., "[1/100]").
                Requires `total` to be provided. Defaults to 0.
            total (int): The total number of items in a batch operation. Used
                for progress display (e.g., "[1/100]"). Requires `index` to be
                provided. Defaults to 0.

        Returns:
            Response: A `Response` object containing the model's output, status,
            duration, and associated messages. Returns a generator if `stream`
            is True.

        Raises:
            ValueError: If neither a prompt nor messages are provided.
            ValueError: If `index` is provided without `total`, or vice-versa.
        """
        # Render our prompt with the input_variables if variables are passed.
        if input_variables and self.prompt:
            logger.info("Rendering prompt with input variables: %s", input_variables)
            prompt = self.prompt.render(input_variables=input_variables)
        elif self.prompt:
            logger.info("Using prompt without input variables.")
            prompt = self.prompt.prompt_string
        else:
            logger.info("No prompt provided, using None.")
            prompt = None
        # Resolve messages: messagestore or user-provided messages.
        if SyncConduit._message_store:
            if messages is not None:
                raise ValueError(
                    "Both messages and message store are provided, please use only one."
                )
            logger.info("Using message store for messages.")
            messages = SyncConduit._message_store.messages
        # Coerce messages and query_input into a list of Message objects
        logger.info("Coercing messages and prompt into a list of Message objects.")
        messages = self._coerce_messages_and_prompt(prompt=prompt, messages=messages)
        assert len(messages) > 0, "No messages provided, cannot run conduit."
        # Route input; if string, if message
        logger.info("Querying model.")
        # Save this for later
        user_message = messages[-1]
        result = self.model.query(
            query_input=messages,
            response_model=self.parser.pydantic_model if self.parser else None,
            verbose=verbose,
            cache=cache,
            index=index,
            total=total,
            stream=stream,
        )
        logger.info(f"Model query completed, return type: {type(result)}.")
        # Save to messagestore if we have one and if we have a response.
        if SyncConduit._message_store:
            if isinstance(result, Response):
                logger.info("Saving response to message store.")
                SyncConduit._message_store.append(result.message)
            elif isinstance(result, ConduitError):
                logger.error("ConduitError encountered, not saving to message store.")
                SyncConduit._message_store.query_failed()  # Remove the last message if it was a query failure.
        if not isinstance(result, ConduitResult):
            logger.warning(
                "Result is not a Response or ConduitError: type {type(result)}."
            )
        return result

    def _coerce_messages_and_prompt(
        self, prompt: str | Message | None, messages: Messages | list[Message] | None
    ) -> list[Message] | Messages:
        """
        We want a list of messages to submit to Model.query.
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
