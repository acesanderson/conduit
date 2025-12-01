from conduit.conduit.conduit_base import ConduitBase
from conduit.result.result import ConduitResult
from conduit.result.response import Response
from conduit.result.error import ConduitError
from conduit.parser.parser import Parser
from conduit.progress.verbosity import Verbosity
from conduit.message.messages import MessageUnion
from conduit.message.message import Message
from conduit.message.textmessage import TextMessage
import logging

logger = logging.getLogger(__name__)


class SyncConduit(ConduitBase):
    def run(
        self,
        # Inputs
        input_variables: dict[str, str] | None = None,
        messages: list[MessageUnion] | None = None,
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
            self.prompt.validate_input_variables(input_variables)
            logger.info("Rendering prompt with input variables: %s", input_variables)
            prompt = self.prompt.render(input_variables=input_variables)
        elif self.prompt:
            logger.info("Using prompt without input variables.")
            prompt = self.prompt.prompt_string
        else:
            logger.info("No prompt provided, using None.")
            prompt = None
        # Resolve messages: messagestore or user-provided messages. Are we including history?
        if self.message_store and include_history:
            if messages is not None:
                raise ValueError(
                    "Both messages and message store are provided, please use only one."
                )
            logger.info("Using message store for messages.")
            messages = self.message_store.messages
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
        if self.message_store and save:
            if isinstance(result, Response):
                logger.info("Saving response to message store.")
                if not include_history:
                    # For a one-off query, the user message that initiated it was not
                    # in the message store. We must add it now to maintain history integrity.
                    user_message = messages[
                        -1
                    ]  # The user message is the last in the list sent to the model
                    self.message_store.append(user_message)
                self.message_store.append(result.message)
                # Context window information
                self.message_store.input_tokens += result.input_tokens
                self.message_store.output_tokens += result.output_tokens
                self.message_store.last_used_model_name = self.model.model
            elif isinstance(result, ConduitError):
                logger.error("ConduitError encountered, not saving to message store.")
                self.message_store.query_failed()
        else:
            if not self.message_store:
                logger.info("No message store associated with conduit, skipping save.")
            if not save:
                logger.info("'save' is False, skipping save to message store.")
        if not isinstance(result, ConduitResult):
            logger.warning(
                "Result is not a Response or ConduitError: type {type(result)}."
            )
        return result

    def _coerce_messages_and_prompt(
        self,
        prompt: str | MessageUnion | None,
        messages: list[MessageUnion] | None,
    ) -> list[MessageUnion]:
        """
        We want a list of messages to submit to ModelSync.query.
        If we have a prompt, we want to convert it into a user message and append it to messages.
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
