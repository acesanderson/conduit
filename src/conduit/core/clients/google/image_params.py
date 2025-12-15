from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class GoogleImageModel(str, Enum):
    """Available Google image generation models"""
    IMAGEN_3_FAST = "imagen-3.0-fast-generate-001"
    IMAGEN_3 = "imagen-3.0-generate-002"


class GoogleImageResponseFormat(str, Enum):
    """Response format for images"""
    URL = "url"
    B64_JSON = "b64_json"


class GoogleImageParams(BaseModel):
    """Parameters for Google image generation (Imagen)"""

    model: GoogleImageModel = Field(
        default=GoogleImageModel.IMAGEN_3,
        description="Imagen model to use"
    )
    response_format: GoogleImageResponseFormat = Field(
        default=GoogleImageResponseFormat.B64_JSON,
        description="Response format"
    )
    n: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4)"
    )
