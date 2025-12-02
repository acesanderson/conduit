from pydantic import BaseModel, Field
import time
from enum import Enum


class StopReason(str, Enum):
    STOP = "stop"  # Natural completion
    LENGTH = "length"  # Hit max_tokens (Context truncation risk!)
    TOOL_CALLS = "tool_calls"  # Model wants to act
    CONTENT_FILTER = "content_filter"  # Safety refusal
    ERROR = "error"  # API error or unknown


class ResponseMetadata(BaseModel):
    timestamp: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Timestamp in milliseconds since epoch",
    )
    duration: float = Field(
        ..., description="Time taken to process the request in milliseconds"
    )
    model_slug: str = Field(..., description="Identifier of the model used")
    input_tokens: int = Field(..., description="Number of input tokens processed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    stop_reason: str | StopReason = Field(
        ..., description="Reason for stopping the generation"
    )
