from Chain.message.message import Message, MessageType, Role
from Chain.message.imagemessage import OpenAITextContent
from Chain.logs.logging_config import get_logger
from pydantic import BaseModel, Field
from typing import Literal, override
from pathlib import Path
import base64, re

logger = get_logger(__name__)


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


class OpenAIInputAudio(BaseModel):
    """
    We are using Gemini through the OpenAI SDK, so we need to define the input audio format.
    Gemini usually supports a range of audio filetypes, but when used with OpenAI SDK, it's only mp3 and wav.
    """

    data: str = Field(description="The base64-encoded audio data.")
    format: Literal["mp3", "wav", ""] = Field(
        description="The format of the audio data, must be 'mp3' or 'wav'."
    )


class OpenAIAudioContent(BaseModel):
    """
    Gemini AudioContent should have a single AudioContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.
    """

    input_audio: OpenAIInputAudio = Field(
        description="The input audio data, must be a base64-encoded string with format 'mp3' or 'wav'."
    )
    type: str = Field(
        default="input_audio", description="The type of content, must be 'input_audio'."
    )


class OpenAIAudioMessage(BaseModel):
    """
    Gemini AudioMessage should have a single AudioContent and a single TextContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.
    """

    role: Role = Field(default="user", description="The role of the message sender.")
    content: list[OpenAIAudioContent | OpenAITextContent]  # type: ignore


class AudioMessage(Message):
    """
    AudioMessage is a message that contains audio content.
    It can be created from an audio file and contains both the audio content in base64 format
    """

    message_type: MessageType = Field(default="audio", exclude=True, repr=False)
    content: list[str] = Field(default_factory=list)
    text_content: str = Field(exclude=True, repr=False)
    audio_content: str = Field(exclude=True, repr=False)
    format: Literal["wav", "mp3"] = Field(exclude=True, repr=False)

    @classmethod
    def from_audio_file(
        cls, audio_file: str | Path, text_content: str, role: str = "user"
    ) -> "AudioMessage":
        """
        Create AudioMessage from audio file.

        Args:
            audio_file (str | Path): Path to the audio file.
            text_content (str): Text content associated with the audio.
            role (str): Role of the message sender, default is "user".
        """
        # Do all the file processing here
        audio_file = Path(audio_file)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file {audio_file} does not exist.")

        audio_content = cls._convert_audio_to_base64(audio_file)
        format = audio_file.suffix.lower()[1:]  # "mp3" or "wav"

        return cls(
            role=role,
            content=[audio_content, text_content],
            text_content=text_content,
            audio_content=audio_content,
            format=format,
        )

    @classmethod
    def from_base64(
        cls,
        audio_content: str,
        text_content: str,
        format: str = "mp3",
        role: str = "user",
    ) -> "AudioMessage":
        """
        Create ImageMessage from base64 image data.

        Args:
            audio_content: Base64-encoded audio data
            text_content: Text prompt/question about the audio (empty string for tts)
            format: either "mp3" or "wav"
            role: Message role (default: "user")

        Returns:
            AudioMessage with processed audio content.
        """
        # Validate base64
        if not is_base64_simple(audio_content):
            raise ValueError("Invalid base64 image data")
        # TBD: optional conversion logic
        # Construct the AudioMessage object
        return cls(
            text_content=text_content,
            audio_content=audio_content,
            format=format,
            role=role,
            content=[audio_content, text_content],
        )

    @classmethod
    def _convert_audio_to_base64(cls, file_path: Path) -> str:
        """
        Convert the audio file to base64 string.
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def __repr__(self):
        """
        String representation of the AudioMessage.
        """
        return f"AudioMessage(role={self.role}, content=[{self.text_content}, {self.audio_content[:30]}...], format={self.format})"

    def play(self):
        """
        Play the audio from the base64 content (no file required).
        """
        from pydub import AudioSegment
        from pydub.playback import play
        import base64
        import io

        # Decode base64 to bytes
        audio_bytes = base64.b64decode(self.audio_content)

        # Create a file-like object from bytes
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio from the buffer
        audio = AudioSegment.from_file(audio_buffer, format=self.format)

        # Play the audio
        play(audio)  # Serialization methods

    @override
    def to_cache_dict(self) -> dict:
        """
        Convert the AudioMessage to a dictionary for caching.
        """
        return {
            "message_type": self.message_type,
            "role": self.role,
            "content": self.content,
            "text_content": self.text_content,
            "audio_content": self.audio_content,
            "format": self.format,
        }

    @override
    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "AudioMessage":
        """
        Create an AudioMessage from a cached dictionary.
        """
        return cls(
            role=cache_dict["role"],
            content=cache_dict["content"],
            text_content=cache_dict["text_content"],
            audio_content=cache_dict["audio_content"],
            format=cache_dict["format"],
        )

    # API compatibility methods
    @override
    def to_openai(self) -> dict:
        """
        Converts the AudioMessage to the OpenAI format.
        """
        openaiinputaudio = OpenAIInputAudio(data=self.audio_content, format=self.format)
        openaiaudiocontent = OpenAIAudioContent(input_audio=openaiinputaudio)
        text_content = OpenAITextContent(text=self.text_content)
        openaiaudiomessage = OpenAIAudioMessage(
            role=self.role, content=[text_content, openaiaudiocontent]
        )
        return openaiaudiomessage.model_dump()

    @override
    def to_anthropic(self):
        raise NotImplementedError("Anthropic API does not support audio messages.")

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
    def to_perplexity(self):
        raise NotImplementedError("Perplexity API does not support audio messages.")
