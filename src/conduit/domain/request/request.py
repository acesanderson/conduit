"""
Model and client interaction:
- Model sends a Request, which is: conversation (list[Message]) + generation_params
- Request sends Response, which is: the request (list[Message]) + generation_params + the assistant message

Use MessageUnion (not Message) because it's a discriminated union.
"""

from __future__ import annotations
from pydantic import BaseModel
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.message.message import Message
from conduit.utils.progress.verbosity import Verbosity
import hashlib
import json
import logging
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.config.conduit_options import ConduitOptions

logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    """
    Inherits all params (temp, top_p) and adds required transport fields.
    """

    messages: list[Message]
    params: GenerationParams

    # Request params
    use_cache: bool | None = True  # Technically: "if cache exists, use it"
    include_history: bool = True  # Whether to include conversation history
    verbosity: Verbosity = Verbosity.PROGRESS

    @classmethod
    def from_conversation(
        cls,
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> GenerationRequest:
        """
        Create a Request from a Conversation, GenerationParams, and ConduitOptions.
        1:1 mapping of fields from Conversation to Request.
        ConduitOptions contains some unserializable fields (i.e. repository), so we only extract relevant options.
        """
        return cls(
            messages=conversation.messages,
            params=params,
            use_cache=options.use_cache,
            include_history=options.include_history,
            verbosity=options.verbosity,
        )

    def generate_cache_key(self) -> str:
        """
        Generate a deterministic SHA256 hash.
        Excludes volatile metadata (timestamps) to ensure semantic caching.
        """
        # Exclude timestamps from all messages to ensure stable hashing
        # This assumes 'timestamp' is the field name in Message
        exclusions = {
            "messages": {
                "__all__": {
                    "timestamp": True,
                    "tool_call_id": True,
                    "tool_calls": {"__all__": {"id": True}},
                },
            },
            "verbosity": True,
            "cache": True,
        }

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
