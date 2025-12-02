"""
Model and client interaction:
- Model sends a Request, which is: conversation (list[Message]) + generation_params
- Request sends Response, which is: the request (list[Message]) + generation_params + the assistant message
"""

from __future__ import annotations
from pydantic import BaseModel
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.output_type import OutputType
from conduit.domain.message.message import Message
import hashlib
import json
import logging
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation

logger = logging.getLogger(__name__)


class Request(BaseModel):
    """
    Inherits all params (temp, top_p) and adds required transport fields.
    """

    output_type: OutputType = "text"  # Routes to different client logic
    params: GenerationParams
    messages: list[Message]

    def generate_cache_key(self) -> str:
        """
        Generate a deterministic SHA256 hash.
        Excludes volatile metadata (timestamps) to ensure semantic caching.
        """
        # Exclude timestamps from all messages to ensure stable hashing
        # This assumes 'timestamp' is the field name in MessageBase
        exclusions = {"messages": {"__all__": {"timestamp": True}}}

        data = self.model_dump(mode="json", exclude_none=True, exclude=exclusions)

        # sort_keys=True is the secret sauce for deterministic JSON
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @property
    def conversation(self) -> Conversation:
        """
        Convert the Request's messages into a Conversation object.
        """
        from conduit.domain.conversation.conversation import Conversation

        return Conversation(messages=self.messages)

    @override
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the Request instance.
        """
        return (
            f"Request(model={self.params.model!r}, messages={self.messages!r}, "
            f"temperature={self.params.temperature!r},"
            f"client_params={self.params.client_params!r}, response_model={self.params.response_model!r})"
        )
