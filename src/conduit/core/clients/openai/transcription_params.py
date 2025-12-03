from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class OpenAITranscriptionModel(str, Enum):
    """Available Whisper models"""

    WHISPER_1 = "whisper-1"


class OpenAITranscriptionResponseFormat(str, Enum):
    """Response formats for transcription"""

    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


class OpenAITranscriptionParams(BaseModel):
    """Parameters for OpenAI audio transcription (Whisper)"""

    model: OpenAITranscriptionModel = Field(
        default=OpenAITranscriptionModel.WHISPER_1, description="Whisper model to use"
    )
    language: str | None = Field(
        default=None, description="ISO-639-1 language code (e.g., 'en', 'es', 'fr')"
    )
    prompt: str | None = Field(
        default=None,
        description="Optional text to guide the model's style or continue a previous segment",
    )
    response_format: OpenAITranscriptionResponseFormat = Field(
        default=OpenAITranscriptionResponseFormat.JSON,
        description="Format of the transcript output",
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature (0-1)"
    )
