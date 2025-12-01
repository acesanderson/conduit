from conduit.model.clients.openai.payload import OpenAIPayload
from typing import Literal


class PerplexityPayload(OpenAIPayload):
    return_citations: bool | None = None
    return_images: bool | None = None
    search_recency_filter: Literal["month", "week", "day", "hour"] | None = None
