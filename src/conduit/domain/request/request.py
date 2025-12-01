"""
Think of it this way:
messages + generation_params  -> Request

Such that:
conduit.params + conversation.messages -> Request
"""

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from conduit.config import settings
from conduit.domain.request.generation_params import GenerationParams
from conduit.utils.progress.display_mixins import (
    RichDisplayParamsMixin,
    PlainDisplayParamsMixin,
)
from conduit.domain.request.outputtype import OutputType
from typing import Any
import logging

logger = logging.getLogger(__name__)


class Request(GenerationParams, RichDisplayParamsMixin, PlainDisplayParamsMixin):
    """
    Inherits all params (temp, top_p) and adds required transport fields.
    """

    messages: list[Message]
    tools: list[dict] | None = None
    response_model: type[BaseModel] | None = Field(default=None, exclude=True)
    client_params: dict[str, Any] | None = None

    # Generated params
    response_model_schema: dict[str, str] | None = None

    @model_validator(mode="after")
    def _populate_schema(self) -> Request:
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

    def generate_cache_key(self) -> str:
        """
        Generate a deterministic SHA256 hash of the request parameters.
        This allows the Cache system to key off this object.
        """
        import json
        import hashlib

        data = self.model_dump(mode="json", exclude_none=True)
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the Request instance.
        """
        return (
            f"Request(model={self.model!r}, messages={self.messages!r}, "
            f"temperature={self.temperature!r},"
            f"client_params={self.client_params!r}, response_model={self.response_model!r})"
        )
