from conduit.utils.progress.display_mixins import (
    RichDisplayConduitErrorMixin,
    PlainDisplayConduitErrorMixin,
)
from typing import Optional, Any
from pydantic import BaseModel, Field
import traceback
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ErrorInfo(BaseModel):
    """Simple error information"""

    code: str = Field(
        ..., description="Error code like 'validation_error', 'api_error', etc."
    )
    message: str = Field(..., description="Human-readable error message")
    category: str = Field(
        ..., description="Error category: 'client', 'server', 'network', 'parsing'"
    )
    timestamp: datetime = Field(..., description="When the error occurred")

    model_config = {"frozen": True}  # Equivalent to dataclass(frozen=True)


class ErrorDetail(BaseModel):
    """Detailed error information for debugging"""

    exception_type: str = Field(
        ..., description="Type of exception like 'ValidationError', 'APIException'"
    )
    stack_trace: Optional[str] = Field(
        None, description="Full stack trace if available"
    )
    raw_response: Optional[Any] = Field(
        None, description="Original response that caused error"
    )
    request_params: Optional[dict] = Field(None, description="Params that led to error")
    retry_count: Optional[int] = Field(None, description="If retries were attempted")

    model_config = {"frozen": True}  # Equivalent to dataclass(frozen=True)


class ConduitError(
    BaseModel, RichDisplayConduitErrorMixin, PlainDisplayConduitErrorMixin
):
    """
    An unsuccessful Result.
    Complete error information.
    We mixin our display classes to provide to_plain and to_rich methods in service of our Progress / Verbosity implementation.
    """

    info: ErrorInfo = Field(..., description="Core error information")
    detail: Optional[ErrorDetail] = Field(
        None, description="Detailed debugging information"
    )

    def __str__(self) -> str:
        """
        Print this like a normal error message + stack trace if available.
        """
        base_message = f"[{self.info.timestamp}] {self.info.category.upper()} - {self.info.code}: {self.info.message}"
        if self.detail and self.detail.stack_trace:
            return f"{base_message}\nStack Trace:\n{self.detail.stack_trace}"
        return base_message

    @classmethod
    def from_exception(
        cls, exc: Exception, code: str, category: str, **context
    ) -> "ConduitError":
        """Create ConduitError from an exception with full context"""
        info = ErrorInfo(
            code=code, message=str(exc), category=category, timestamp=datetime.now()
        )

        detail = ErrorDetail(
            exception_type=type(exc).__name__,
            stack_trace=traceback.format_exc(),
            raw_response=context.get("raw_response"),
            request_params=context.get("request_params"),
            retry_count=context.get("retry_count"),
        )

        return cls(info=info, detail=detail)

    @classmethod
    def simple(cls, code: str, message: str, category: str) -> "ConduitError":
        """Create simple error without exception details"""
        info = ErrorInfo(
            code=code, message=message, category=category, timestamp=datetime.now()
        )
        return cls(info=info, detail=None)

    def to_cache_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "ConduitError":
        return cls(**cache_dict)
