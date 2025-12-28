"""
Model and client interaction:
- Model sends a Request, which is: conversation (list[MessageUnion]) + generation_params
- Request sends Response, which is: the request (list[MessageUnion]) + generation_params + the assistant message

Use MessageUnion (not Message) because it's a discriminated union.
"""

from __future__ import annotations
from pydantic import BaseModel
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.message.message import MessageUnion
from conduit.utils.progress.verbosity import Verbosity
from collections.abc import Sequence
import hashlib
import json
import logging
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from conduit.domain.message.message import Message
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.request.query_input import QueryInput

logger = logging.getLogger(__name__)


# Hashing helpers
def _canonical_json_bytes(obj: object) -> bytes:
    """
    Canonical JSON encoding for hashing:
    - sorted keys for deterministic order
    - compact separators to avoid whitespace differences
    - UTF-8 bytes for stable hashing
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _response_model_id(model_cls: type[BaseModel] | None) -> str | None:
    """
    Stable identifier for a Pydantic model class used for structured responses.
    Uses module + qualname to avoid collisions between same-named classes.
    """
    if model_cls is None:
        return None
    return f"{model_cls.__module__}.{model_cls.__qualname__}"


class GenerationRequest(BaseModel):
    """
    Inherits all params (temp, top_p) and adds required transport fields.
    """

    messages: Sequence[MessageUnion]
    params: GenerationParams
    options: ConduitOptions

    # Request param overrides
    use_cache: bool | None = True  # Technically: "if cache exists, use it"
    include_history: bool = True  # Whether to include conversation history
    verbosity_override: Verbosity | None = None

    def generate_cache_key(self) -> str:
        """
        Cache identity = canonical(params) + ordered(messages).

        - Params: all GenerationParams fields EXCEPT response_model and response_model_schema,
          plus response_model's module.qualname when present.
        - Messages: ordered list of {role, content} pairs only.
        - Canonical JSON (sorted keys) is hashed with SHA256.
        """
        key_payload = {
            "params": self._normalize_params_for_cache(),
            "messages": self._normalize_messages_for_cache(),
        }
        return hashlib.sha256(_canonical_json_bytes(key_payload)).hexdigest()

    def _normalize_params_for_cache(self) -> dict:
        # Dump JSON-safe primitives deterministically; exclude structured-response internals.
        params_dump = self.params.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "response_model": True,
                "response_model_schema": True,
            },
        )

        # Add the semantic identity of the response model (if any).
        params_dump["response_model"] = _response_model_id(self.params.response_model)
        return params_dump

    def _normalize_messages_for_cache(self) -> list[dict]:
        # Ordered list; each message contributes only role + content.
        return [{"role": m.role_str, "content": m.content} for m in self.messages]

    @classmethod
    def from_conversation(
        cls,
        conversation: Conversation,
        params: GenerationParams,
        options: ConduitOptions,
    ) -> GenerationRequest:
        """
        Create a GenerationRequest from a Conversation, GenerationParams, and ConduitOptions.
        """
        return cls(
            messages=conversation.messages,
            params=params,
            options=options,
        )

    @classmethod
    def from_query_input(
        cls, query_input: QueryInput, params: GenerationParams, options: ConduitOptions
    ) -> GenerationRequest:
        """
        Create a GenerationRequest from a QueryInput object.
        """
        from conduit.domain.request.query_input import constrain_query_input

        messages = constrain_query_input(query_input)

        return cls(
            messages=messages,
            params=params,
            options=options,
        )

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
