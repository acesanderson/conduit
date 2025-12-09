from __future__ import annotations
from conduit.config import settings
from conduit.core.model.model_sync import ModelSync
from conduit.core.prompt.prompt import Prompt
from conduit.core.parser.parser import Parser
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.conversation.conversation import Conversation
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rich.console import Console
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.result.response import Response
    from conduit.core.engine.engine import Engine
    from conduit.storage.repository.protocol import (
        ConversationRepository,
    )
    from conduit.storage.cache.protocol import ConduitCache


class ConduitSync:
    def __init__(
        self,
        # Required
        model: ModelSync,
        # Optional
        project_name: str = "conduit",
        repository: ConversationRepository | None = None,
        prompt: Prompt | None = None,
        parser: Parser | None = None,
    ):
        # Initial attributes
        self.model: ModelSync = model
        self.project_name: str = project_name
        self.repository: ConversationRepository | None = repository
        self.prompt: Prompt | None = prompt
        self.parser: Parser | None = parser
        # Working attributes
        self.conversation: Conversation | None = None

    @classmethod
    def create(
        cls,
        project_name: str,
        model: ModelSync | str,
        prompt: Prompt | str,
        parser: Parser | None = None,
        # Configurations
        persist: bool = False,
        cached: bool = False,
        verbosity: Verbosity = settings.default_verbosity,
        params: dict[str, str | dict[str, str]] | None = None,
    ) -> ConduitSync:
        # Model
        if isinstance(model, str):
            model_instance = ModelSync(model_name=model)
        else:
            model_instance = model
        # Prompt
        if isinstance(prompt, str):
            prompt_instance = Prompt(prompt)
        elif isinstance(prompt, Prompt):
            prompt_instance = prompt
        # Repository
        repository_instance = None
        if persist:
            repository_instance: ConversationRepository = settings.default_repository(
                name=project_name
            )
        # Cache
        cache_instance = None
        if cached:
            cache_instance: ConduitCache = settings.default_cache(name=project_name)
        # Params
        params_instance = None
        if params:
            params_instance = GenerationParams(**params)

        # Build the model first
        model_instance.verbosity = verbosity
        model_instance.cache = cache_instance
        model_instance.params = params_instance

        return cls(
            model=model_instance,
            project_name=project_name,
            repository=repository_instance,
            prompt=prompt_instance,
            parser=parser,
        )

    def run(
        self,
        input_variables: dict[str, str] | None = None,
        stream: bool = False,
        verbosity: Verbosity = settings.default_verbosity,
        # Progress tracking
        index: int = 0,
        total: int = 0,
        # Possible exemptions
        cached: bool = True,
        persist: bool = True,
        include_history: bool = True,
        save: bool = True,
    ) -> Response: ...

    @property
    def cache(self) -> ConduitCache | None:
        return self.model.cache

    @property
    def params(self) -> GenerationParams:
        return self.model.params

    @property
    def console(self) -> Console | None:
        return self.model.console

    @property
    def topic(self) -> str:
        if self.conversation and self.conversation.topic:
            return self.conversation.topic
        else:
            return "Untitled"

    @topic.setter
    def topic(self, value: str) -> None:
        if self.conversation:
            self.conversation.topic = value
        else:
            logger.warning("No conversation exists to set the name on. Name not set.")

    # Conversation management
    def _execute_with_engine(self, conversation: Conversation) -> Conversation:
        """
        Given a conversation, execute it using an Engine.
        If needed, can take decorators for telemetry, logging, progress, caching, etc.
        """
        raise NotImplementedError("_execute_with_engine not yet implemented.")

    def _update_topic(self) -> str:
        """
        Create a title for the current conversation based on the prompt.
        Will leverage an external service.
        """
        raise NotImplementedError("_create_title not yet implemented.")


#         if input_variables and self.prompt:
#             self.prompt.validate_input_variables(input_variables)
#             logger.info("Rendering prompt with input variables: %s", input_variables)
#             prompt = self.prompt.render(input_variables=input_variables)
#         elif self.prompt:
#             logger.info("Using prompt without input variables.")
#             prompt = self.prompt.prompt_string
#         else:
#             logger.info("No prompt provided, using None.")
#             prompt = None
#         # Resolve messages: messagestore or user-provided messages. Are we including history?
#         if self.message_store and include_history:
#             if messages is not None:
#                 raise ValueError(
#                     "Both messages and message store are provided, please use only one."
#                 )
#             logger.info("Using message store for messages.")
#             messages = self.message_store.messages
#         else:
#             if messages is None:
#                 logger.info("No message store or messages provided, starting fresh.")
#                 messages = []
#             else:
#                 logger.info("Using user-provided messages.")
#         # Coerce messages and query_input into a list of Message objects
#         logger.info("Coercing messages and prompt into a list of Message objects.")
#         messages = self._coerce_messages_and_prompt(prompt=prompt, messages=messages)
#         assert len(messages) > 0, "No messages provided, cannot run conduit."
#         # Route input; if string, if message
#         logger.info("Querying model.")
#         # Save this for later
#         result = self.model.query(
#             query_input=messages,
#             response_model=self.parser.pydantic_model if self.parser else None,
#             verbose=verbose,
#             cache=cache,
#             index=index,
#             total=total,
#             stream=stream,
#         )
#         logger.info(f"ModelSync query completed, return type: {type(result)}.")
#         # Save to messagestore if we have one and if we have a response.
#         if self.message_store and save:
#             if isinstance(result, Response):
#                 logger.info("Saving response to message store.")
#                 if not include_history:
#                     # For a one-off query, the user message that initiated it was not
#                     # in the message store. We must add it now to maintain history integrity.
#                     user_message = messages[
#                         -1
#                     ]  # The user message is the last in the list sent to the model
#                     self.message_store.append(user_message)
#                 self.message_store.append(result.message)
#                 # Context window information
#                 self.message_store.input_tokens += result.input_tokens
#                 self.message_store.output_tokens += result.output_tokens
#                 self.message_store.last_used_model_name = self.model.model
#         else:
#             if not self.message_store:
#                 logger.info("No message store associated with conduit, skipping save.")
#             if not save:
#                 logger.info("'save' is False, skipping save to message store.")
#         if not isinstance(result, ConduitResult):
#             logger.warning("Result is not a Response: type {type(result)}.")
#         return result
#
#     def _coerce_messages_and_prompt(
#         self,
#         prompt: str | Message | None,
#         messages: list[Message] | None,
#     ) -> list[Message]:
#         """
#         We want a list of messages to submit to ModelSync.query.
#         If we have a prompt, we want to convert it into a user message and append it to messages.
#         """
#         if not messages:
#             messages = []
#         if isinstance(prompt, Message):
#             # If we have a message, just append it
#             messages.append(prompt)
#         elif isinstance(prompt, str):
#             # If we have a string, convert it to a TextMessage
#             messages.append(UserMessage(content=prompt))
#         elif prompt is None:
#             # If we have no prompt, do nothing
#             pass
#         else:
#             raise ValueError(f"Unsupported query_input type: {type(query_input)}")
#
#         return messages
