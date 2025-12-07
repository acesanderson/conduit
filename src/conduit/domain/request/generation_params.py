"""
TO IMPLEMENT:
- params should cascade: Conduit defaults < Conversation overrides < Request final
"""

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
import logging

logger = logging.getLogger(__name__)


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
    client_params: dict | None = None

    # For structured responses; excluded from serialization, trust me
    response_model: type[BaseModel] | None = Field(default=None, exclude=True)
    # Generated field
    response_model_schema: dict[str, str] | None = None

    @model_validator(mode="after")
    def _populate_schema(self) -> GenerationParams:
        """
        If a Pydantic class is provided in response_model but no schema is set,
        generate the schema automatically. This ensures the request is cacheable/serializable.
        """
        if self.response_model and not self.response_model_schema:
            try:
                self.response_model_schema = self.response_model.model_json_schema()
            except AttributeError:
                logger.warning(f"Could not generate schema for {self.response_model}")
        return self
