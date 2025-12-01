from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    """
    Standard tunable parameters for LLM inference.
    Shared by Conduit (defaults), Conversation (overrides), and Request (final payload).
    """

    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] | None = None
    stream: bool = False
