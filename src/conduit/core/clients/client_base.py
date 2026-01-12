"""
Base class for clients; openai, anthropic, etc. inherit from this class.
Both sync and async methods are defined here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, override
import logging

if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.core.clients.payload_base import Payload
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.message.message import Message


logger = logging.getLogger(__name__)


class Client:
    def _convert_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        return [self._convert_message(m) for m in messages]

    def _convert_message(self, message: Message) -> dict[str, Any]:
        raise NotImplementedError("Should be implemented in subclass.")

    def _convert_request(self, request: GenerationRequest) -> Payload:
        raise NotImplementedError("Should be implemented in subclass.")

    async def query(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError("Should be implemented in subclass.")

    async def tokenize(self, model: str, payload: str | Sequence[Message]) -> int:
        raise NotImplementedError("Should be implemented in subclass.")

    @override
    def __repr__(self):
        """
        Standard repr.
        """
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
