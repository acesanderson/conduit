"""
There are two basic message formats:
1. OpenAI (applicable to ollama, groq, gemini, etc.)
2. Anthropic (the one hold out)

We have a basic ImageMessage class, which is a wrapper for the OpenAI and Anthropic formats.
"""

from pydantic import BaseModel, Field
from conduit.domain.message.message import Message, MessageType, Role
from conduit.domain.message.convert_image import convert_image, convert_image_file
from pathlib import Path
from typing import override
import re
import logging

logger = logging.getLogger(__name__)

# Map PIL formats to MIME types
format_to_mime = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


def extension_to_mimetype(image_file: Path) -> str:
    """
    Given a Path object, return the mimetype.
    """
    extension = image_file.suffix.lower()
    try:
        mimetype = format_to_mime[extension]
        return mimetype
    except:
        raise ValueError(
            f"Unsupported image format: {extension}. Supported formats are: {', '.join(format_to_mime.keys())}"
        )


# Provider-specific classes -- these are NOT serialized at all, but constructed.
class AnthropicTextContent(BaseModel):
    type: str = "text"
    text: str


class AnthropicImageSource(BaseModel):
    type: str = "base64"
    media_type: str
    data: str


class AnthropicImageContent(BaseModel):
    type: str = "image"
    source: AnthropicImageSource


class AnthropicImageMessage(BaseModel):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.
    """

    role: Role = Field(default="user", description="The role of the message sender.")
    content: list[AnthropicImageContent | AnthropicTextContent]  # type: ignore


# OpenAI-specific message classes
class OpenAITextContent(BaseModel):
    type: str = "text"
    text: str = Field(description="The text content of the message, i.e. the prompt.")


class OpenAIImageUrl(BaseModel):
    """Nested object for OpenAI image URL structure"""

    url: str = Field(description="The data URL with base64 image")


class OpenAIImageContent(BaseModel):
    """
    OpenAI requires image_url to be an object, not a string
    """

    type: str = "image_url"
    image_url: OpenAIImageUrl = Field(description="The image URL object")


class OpenAIImageMessage(BaseModel):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.
    """

    role: Role = Field(default="user", description="The role of the message sender.")
    content: list[OpenAIImageContent | OpenAITextContent]  # type: ignore


# Our base ImageMessage class with serialization support
class ImageMessage(Message):
    """
    ImageMessage with serialization/deserialization support.

    Create with:
    1. ImageMessage.from_image_file(image_file, text_content) - from file
    2. ImageMessage(role="user", content=[image_data, text], ...) - from processed data

    You can convert it to provider formats with to_openai() and to_anthropic() methods.
    """

    message_type: MessageType = Field(default="image", exclude=True, repr=False)
    content: list[str] = Field(
        default_factory=list, description="[image_content, text_content]"
    )
    text_content: str = Field(
        description="The text content/prompt", exclude=True, repr=False
    )
    image_content: str = Field(
        description="Base64-encoded PNG image", exclude=True, repr=False
    )
    mime_type: str = Field(
        description="Always 'image/png' after processing",
        default="image/png",
        exclude=True,
        repr=False,
    )

    @classmethod
    def from_image_file(
        cls, image_file: str | Path, text_content: str, role: str = "user"
    ) -> "ImageMessage":
        """
        Create ImageMessage from image file.

        Args:
            image_file: Path to image file (any supported format)
            text_content: Text prompt/question about the image
            role: Message role (default: "user")

        Returns:
            ImageMessage with processed image data

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is unsupported
        """
        image_path = Path(image_file)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist")

        # Validate file format
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")

        # Convert image to optimized PNG base64
        try:
            image_content = convert_image_file(image_path)
        except Exception as e:
            raise ValueError(f"Failed to process image file {image_path}: {e}")

        # Validate the conversion worked
        if not is_base64_simple(image_content):
            raise ValueError("Image conversion produced invalid base64 data")

        return cls(
            role=role,
            content=[image_content, text_content],
            text_content=text_content,
            image_content=image_content,
            mime_type="image/png",  # Always PNG after convert_image_file()
        )

    @classmethod
    def from_base64(
        cls,
        image_content: str,
        text_content: str,
        mime_type: str = "image/png",
        role: str = "user",
    ) -> "ImageMessage":
        """
        Create ImageMessage from base64 image data.

        Args:
            image_content: Base64-encoded image data
            text_content: Text prompt/question about the image
            mime_type: MIME type of the image (will be converted to PNG)
            role: Message role (default: "user")

        Returns:
            ImageMessage with processed image data
        """
        # Validate base64
        if not is_base64_simple(image_content):
            raise ValueError("Invalid base64 image data")

        # Convert to PNG if needed
        if mime_type != "image/png":
            try:
                image_content = convert_image(image_content)
                mime_type = "image/png"
            except Exception as e:
                raise ValueError(f"Failed to convert image to PNG: {e}")

        return cls(
            role=role,
            content=[image_content, text_content],
            text_content=text_content,
            image_content=image_content,
            mime_type=mime_type,
        )

    def __repr__(self):
        return f"ImageMessage(role={self.role}, text_content='{self.text_content[:30]}...', mime_type={self.mime_type})"

    def display(self):
        """
        Display a base64-encoded image using chafa.
        Your mileage may vary depending on the terminal and chafa version.
        """
        import subprocess, base64, os

        try:
            image_data = base64.b64decode(self.image_content)
            cmd = ["chafa", "-"]

            # If in tmux or SSH, force text mode for consistency
            if (
                os.environ.get("TMUX")
                or os.environ.get("SSH_CLIENT")
                or os.environ.get("SSH_CONNECTION")
            ):
                cmd.extend(["--format", "symbols", "--symbols", "block"])
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            process.communicate(input=image_data)
        except Exception as e:
            print(f"Error displaying image: {e}")

    # Serialization methods
    @override
    def to_cache_dict(self) -> dict:
        """
        Convert the ImageMessage to a dictionary for caching.

        Returns:
            A dictionary representation of the ImageMessage.
        """
        return {
            "message_type": self.message_type,
            "role": self.role,
            "text_content": self.text_content,
            "image_content": self.image_content,
            "mime_type": self.mime_type,
        }

    @override
    @classmethod
    def from_cache_dict(cls, data: dict) -> "ImageMessage":
        """
        Create an ImageMessage from a cached dictionary.

        Args:
            data: Dictionary containing the cached data.

        Returns:
            An ImageMessage instance.

        Raises:
            ValidationError: If the data is invalid.
        """
        return cls(
            message_type=data["message_type"],
            role=data["role"],
            text_content=data["text_content"],
            image_content=data["image_content"],
            mime_type=data["mime_type"],
        )

    # API compatibility methods. These are overridden in AudioMessage, ImageMessage, and other specialized message types (TBD).
    @override
    def to_openai(self) -> dict:
        """Converts the ImageMessage to the OpenAI format."""
        # Create the nested URL object
        image_url_obj = OpenAIImageUrl(
            url=f"data:{self.mime_type};base64,{self.image_content}"
        )

        # Create image content with the nested object
        image_content = OpenAIImageContent(image_url=image_url_obj)

        # Create text content
        text_content = OpenAITextContent(text=self.text_content)

        openaiimagemessage = OpenAIImageMessage(
            role=self.role,
            content=[text_content, image_content],  # Note: text first, then image
        )
        return openaiimagemessage.model_dump()

    @override
    def to_anthropic(self) -> dict:
        """Converts the ImageMessage to the Anthropic format."""
        image_source = AnthropicImageSource(
            type="base64", media_type=self.mime_type, data=self.image_content
        )
        image_content = AnthropicImageContent(source=image_source)
        text_content = AnthropicTextContent(text=self.text_content)
        anthropicimagemessage = AnthropicImageMessage(
            role=self.role, content=[image_content, text_content]
        )
        return anthropicimagemessage.model_dump()

    @override
    def to_google(self) -> dict:
        """
        Defaults to OpenAI format for Google Gemini.
        """
        return self.to_openai()

    @override
    def to_ollama(self) -> dict:
        """
        Defaults to OpenAI format for Ollama.
        """
        return self.to_openai()

    @override
    def to_perplexity(self) -> dict:
        """
        Convert message to Perplexity API format.
        """
        raise NotImplementedError("Perplexity API does not support image messages.")
