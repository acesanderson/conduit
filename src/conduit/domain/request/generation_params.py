"""
TO IMPLEMENT:
- params should cascade: Conduit defaults < Conversation overrides < Request final
"""

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator, field_validator
from conduit.domain.request.output_type import OutputType
from typing import Any
import logging

logger = logging.getLogger(__name__)


class GenerationParams(BaseModel):
    """
    Standard tunable parameters for LLM inference.
    Shared by Conduit (defaults), Conversation (overrides), and Request (final payload).

    WARNING (cache determinism):
    GenerationParams participates in request cache keys. Any new field added here MUST be:
    - deterministic across runs when serialized (no timestamps, UUIDs, random/default_factory, env-derived values),
    - JSON-serializable via Pydantic (or explicitly normalized/excluded in GenerationRequest._normalize_params_for_cache),
    - and intentionally considered as affecting (or not affecting) cache identity.

    If a new field should NOT affect caching, exclude it explicitly in _normalize_params_for_cache.
    """

    output_type: OutputType = "text"
    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: list[str] | None = None
    stream: bool = False
    client_params: dict | None = None
    system: str | None = None
    tools: list[dict[str, Any]] | None = None

    # For structured responses; excluded from serialization, trust me
    response_model: type[BaseModel] | None = Field(default=None, exclude=True)
    # Generated field
    response_model_schema: dict[str, str] | None = None

    @field_validator("model")
    def _validate_model(cls, v: str) -> str:
        """
        Validate that the model is recognized.
        """
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.validate_model(v)

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

    @classmethod
    def defaults(cls, model_name: str) -> GenerationParams:
        """
        Return default generation parameters.
        """
        return cls(
            model=model_name,
            temperature=0.7,
            top_p=1.0,
            max_tokens=2048,
            stream=False,
            output_type="text",
        )
