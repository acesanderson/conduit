from __future__ import annotations
from conduit.core.clients.openai.payload import OpenAIPayload


class MistralPayload(OpenAIPayload):
    safe_prompt: bool | None = None
    random_seed: int | None = None
    prompt_mode: str | None = None
    prediction: str | None = None
