from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class GoogleImageModel(str, Enum):
    """Available Google image generation models (Gemini API / AI Studio)"""
    GEMINI_FLASH_IMAGE = "gemini-2.5-flash-image"
    GEMINI_PRO_IMAGE = "gemini-3-pro-image-preview"


class GoogleImageParams(BaseModel):
    """Parameters for Google image generation"""

    model: GoogleImageModel = Field(
        default=GoogleImageModel.GEMINI_FLASH_IMAGE,
        description="Image generation model to use"
    )
    n: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4)"
    )
