from conduit.model.clients.openai.payload import OpenAIPayload
from typing import Any


class OllamaPayload(OpenAIPayload):
    extra_body: dict[str, Any] | None = None
