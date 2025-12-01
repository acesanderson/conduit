"""
Base class for clients; openai, anthropic, etc. inherit from this class.
ABC abstract methods are like pydantic validators for classes. They ensure that the child classes implement the methods defined in the parent class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from conduit.core.model.clients.payload_base import Payload
    from conduit.storage.odometer.usage import Usage
    from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
    from conduit.domain.request.request import Request
    from conduit.domain.message.message import Message


class Client(ABC):
    @abstractmethod
    def __init__(self):
        """
        Typically we should see this here:
        self._client = self._initialize_client()
        """
        pass

    @abstractmethod
    def _initialize_client(self) -> object:
        """
        This method should initialize the client object, this is idiosyncratic to each SDK.
        """
        pass

    @abstractmethod
    def _get_api_key(self) -> str:
        """
        API keys are accessed via dotenv.
        This should be in an .env file in the root directory.
        """
        pass

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Helper: maps the list of internal Messages to a list of provider dicts.
        Override this if the provider needs whole-list manipulation.
        """
        return [self._convert_message(m) for m in messages]

    @abstractmethod
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into the provider's specific message format.
        Uses pattern matching on the message type.
        """
        pass

    @abstractmethod
    def _convert_request(self, request: Request) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by this provider's SDK.

        This handles:
        - Message role mapping (e.g. OpenAI 'user' vs Google 'user')
        - Parameter renaming (max_tokens vs max_output_tokens)
        - Specific structural requirements (e.g. Anthropic system prompt extraction)
        """
        pass

    @abstractmethod
    def query(
        self, request: Request
    ) -> tuple[str | BaseModel | SyncStream | AsyncStream, Usage]:
        """
        All client subclasses must have a query function that can take:
        - a Request object, which contains all the parameters needed for the query

        And returns
        - A tuple of
            - either a string (i.e. text generation) or a Pydantic model (function calling)
            - a Usage object containing input and output token counts
        """
        pass

    @abstractmethod
    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Get the token count for a text or a message history.

        Args:
            model: The specific model name (e.g. "gpt-4o", "claude-3-5-sonnet").
            payload:
                - str: Returns the raw token count of the text string (no message overhead).
                - list[Message]: Returns the total token count for a conversation history,
                  including message overhead (roles, start/end tokens, etc).
        """
        pass

    def __repr__(self):
        """
        Standard repr.
        """
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
