from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class OpenAIImageModel(str, Enum):
    """Available DALL-E models"""
    DALL_E_2 = "dall-e-2"
    DALL_E_3 = "dall-e-3"


class OpenAIImageSize(str, Enum):
    """Available image sizes"""
    SIZE_256 = "256x256"      # DALL-E 2 only
    SIZE_512 = "512x512"      # DALL-E 2 only
    SIZE_1024 = "1024x1024"   # Both models
    SIZE_1792_1024 = "1792x1024"  # DALL-E 3 only
    SIZE_1024_1792 = "1024x1792"  # DALL-E 3 only


class OpenAIImageQuality(str, Enum):
    """Image quality (DALL-E 3 only)"""
    STANDARD = "standard"
    HD = "hd"


class OpenAIImageStyle(str, Enum):
    """Image style (DALL-E 3 only)"""
    VIVID = "vivid"
    NATURAL = "natural"


class OpenAIImageResponseFormat(str, Enum):
    """Response format for images"""
    URL = "url"
    B64_JSON = "b64_json"


class OpenAIImageParams(BaseModel):
    """Parameters for OpenAI image generation (DALL-E)"""

    model: OpenAIImageModel = Field(
        default=OpenAIImageModel.DALL_E_3,
        description="DALL-E model to use"
    )
    size: OpenAIImageSize = Field(
        default=OpenAIImageSize.SIZE_1024,
        description="Size of generated image"
    )
    quality: OpenAIImageQuality = Field(
        default=OpenAIImageQuality.STANDARD,
        description="Image quality (DALL-E 3 only)"
    )
    style: OpenAIImageStyle = Field(
        default=OpenAIImageStyle.VIVID,
        description="Image style (DALL-E 3 only)"
    )
    response_format: OpenAIImageResponseFormat = Field(
        default=OpenAIImageResponseFormat.B64_JSON,
        description="Response format"
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of images (DALL-E 2: 1-10, DALL-E 3: only 1)"
    )
