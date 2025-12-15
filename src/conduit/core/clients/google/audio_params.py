from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class GoogleTTSVoice(str, Enum):
    """Available voices for Google Gemini TTS"""
    AOEDE = "aoede"
    CHARLIE = "charlie"
    CHARON = "charon"
    FENRIR = "fenrir"
    KORE = "kore"
    PUCK = "puck"


class GoogleTTSModel(str, Enum):
    """Available Google TTS models"""
    GEMINI_2_5_FLASH_TTS = "gemini-2.5-flash-preview-tts"


class GoogleAudioFormat(str, Enum):
    """Available audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    PCM = "pcm"


class GoogleAudioParams(BaseModel):
    """Parameters for Google audio generation (TTS)"""

    voice: GoogleTTSVoice = Field(
        default=GoogleTTSVoice.AOEDE,
        description="Voice to use for TTS"
    )
    model: GoogleTTSModel = Field(
        default=GoogleTTSModel.GEMINI_2_5_FLASH_TTS,
        description="TTS model to use"
    )
    response_format: GoogleAudioFormat = Field(
        default=GoogleAudioFormat.MP3,
        description="Audio format for output"
    )
