from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class OpenAITTSVoice(str, Enum):
    """Available voices for OpenAI TTS"""

    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"


class OpenAITTSModel(str, Enum):
    """Available TTS models"""

    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    GPT_4O_MINI_TTS = "gpt-4o-mini-tts"


class OpenAIAudioFormat(str, Enum):
    """Available audio formats"""

    MP3 = "mp3"
    WAV = "wav"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    PCM = "pcm"


class OpenAIAudioParams(BaseModel):
    """Parameters for OpenAI audio generation (TTS)"""

    voice: OpenAITTSVoice = Field(
        default=OpenAITTSVoice.ALLOY, description="Voice to use for TTS"
    )
    model: OpenAITTSModel = Field(
        default=OpenAITTSModel.TTS_1, description="TTS model to use"
    )
    response_format: OpenAIAudioFormat = Field(
        default=OpenAIAudioFormat.MP3, description="Audio format for output"
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Speed of speech (0.25-4.0)"
    )
