from __future__ import annotations
from conduit.core.clients.openai.payload import OpenAIPayload
from typing import Any


class LlamaCppPayload(OpenAIPayload):
    top_logprobs: int | None = None
    extra_body: dict[str, Any] | None = None
