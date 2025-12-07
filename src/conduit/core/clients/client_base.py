"""
Base class for clients; openai, anthropic, etc. inherit from this class.
Both sync and async methods are defined here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, override
import logging

if TYPE_CHECKING:
    from conduit.core.clients.payload_base import Payload
    from conduit.domain.request.request import Request
    from conduit.domain.result.result import ConduitResult
    from conduit.domain.message.message import Message


logger = logging.getLogger(__name__)


class Client:
    def __init__(self):
        raise NotImplementedError("Should be implemented in subclass.")

    def _initialize_client(self) -> object:
        raise NotImplementedError("Should be implemented in subclass.")

    def _get_api_key(self) -> str:
        raise NotImplementedError("Should be implemented in subclass.")

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        return [self._convert_message(m) for m in messages]

    def _convert_message(self, message: Message) -> dict[str, Any]:
        raise NotImplementedError("Should be implemented in subclass.")

    def _convert_request(self, request: Request) -> Payload:
        raise NotImplementedError("Should be implemented in subclass.")

    def query(self, request: Request) -> ConduitResult:
        raise NotImplementedError("Should be implemented in subclass.")

    async def query_async(self, request: Request) -> ConduitResult:
        raise NotImplementedError(
            "Should be implemented in subclass if async is supported."
        )

    def tokenize(self, model: str, payload: str | list[Message]) -> int:
        raise NotImplementedError("Should be implemented in subclass.")

    async def tokenize_async(self, model: str, payload: str | list[Message]) -> int:
        raise NotImplementedError(
            "Should be implemented in subclass if async is supported."
        )

    @override
    def __repr__(self):
        """
        Standard repr.
        """
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
