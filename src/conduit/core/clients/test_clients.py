from __future__ import annotations
from conduit.core.model.clients.openai.client import OpenAIClientSync, OpenAIClientAsync
from conduit.domain.request.request import Request
from conduit.domain.result.response import Response
from conduit.domain.message.message import (
    UserMessage,
    TextContent,
    AudioContent,
    ImageContent,
)
from conduit.domain.request.generation_params import GenerationParams
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
from pydantic import BaseModel
import pytest
import base64
from pathlib import Path

AUDIO_PATH = (
    Path.home()
    / "Brian_Code"
    / "conduit-project"
    / "src"
    / "conduit"
    / "core"
    / "model"
    / "clients"
    / "harvard.mp3"
)
assert AUDIO_PATH.exists(), f"Audio file not found at {AUDIO_PATH}"
IMAGE_PATH = (
    Path.home()
    / "Brian_Code"
    / "conduit-project"
    / "src"
    / "conduit"
    / "core"
    / "model"
    / "clients"
    / "image.png"
)
assert IMAGE_PATH.exists(), f"Image file not found at {IMAGE_PATH}"


class Frog(BaseModel):
    """Test model for structured responses"""

    species: str
    age: int
    name: str
    occupation: str
    color: str
    legs: str
    continent: str


# OpenAI fixtures
@pytest.fixture
def openai_text_request():
    """Basic text generation request"""
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    params = GenerationParams(model=model)
    return Request(params=params, messages=messages)


@pytest.fixture
def openai_stream_request():
    """Streaming text generation request"""
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    params = GenerationParams(model=model, stream=True)
    return Request(params=params, messages=messages)


@pytest.fixture
def openai_structured_request():
    """Structured response request"""
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    params = GenerationParams(model=model, response_model=Frog)
    return Request(params=params, messages=messages)


# Helper functions for multimodal tests
def load_audio_as_base64(audio_path: Path) -> str:
    """Load an audio file and convert to base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_image_as_base64(image_path: Path) -> str:
    """Load an image file and convert to base64 data URL"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    # Determine image format from extension
    ext = image_path.suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{image_data}"


@pytest.fixture
def openai_audio_input_request():
    """Audio input request using harvard.mp3"""
    # Load the audio file
    audio_base64 = load_audio_as_base64(AUDIO_PATH)

    # Create multimodal content with text and audio
    text_content = TextContent(text="What does this audio say?")
    audio_content = AudioContent(data=audio_base64, format="mp3")

    user_message = UserMessage(content=[text_content, audio_content])
    messages = [user_message]

    # Use the audio-capable model
    model = "gpt-4o-audio-preview"
    params = GenerationParams(model=model)
    return Request(params=params, messages=messages, output_type="text")


@pytest.fixture
def openai_audio_generation_request():
    """Audio generation (TTS) request"""
    user_message = UserMessage(content="Hello, this is a test of text to speech.")
    messages = [user_message]

    # Default TTS model
    model = "tts-1"
    params = GenerationParams(model=model)
    return Request(params=params, messages=messages, output_type="audio")


@pytest.fixture
def openai_audio_transcription_request():
    """Audio transcription (Whisper) request using harvard.mp3"""
    # Load the audio file
    audio_base64 = load_audio_as_base64(AUDIO_PATH)

    # Create audio content
    audio_content = AudioContent(data=audio_base64, format="mp3")

    user_message = UserMessage(content=[audio_content])
    messages = [user_message]

    # Use the Whisper model
    model = "whisper-1"
    params = GenerationParams(model=model)
    return Request(params=params, messages=messages, output_type="transcription")


@pytest.fixture
def openai_image_analysis_request():
    """Image analysis (vision) request using image.png"""
    # Load the image file
    image_path = Path(__file__).parent / "image.png"
    image_url = load_image_as_base64(image_path)

    # Create multimodal content with text and image
    text_content = TextContent(text="What is in this image? Describe it in detail.")
    image_content = ImageContent(url=image_url, detail="high")

    user_message = UserMessage(content=[text_content, image_content])
    messages = [user_message]

    # Use the vision-capable model
    model = "gpt-4o"
    params = GenerationParams(model=model)
    return Request(params=params, messages=messages, output_type="text")


@pytest.fixture
def openai_image_generation_request():
    """Image generation (DALL-E) request"""
    user_message = UserMessage(
        content="A serene landscape with mountains and a lake at sunset"
    )
    messages = [user_message]

    # Default DALL-E model
    model = "dall-e-3"
    params = GenerationParams(model=model)
    return Request(params=params, messages=messages, output_type="image")


####################
# OpenAI Tests
####################


def test_openai_text_generation_sync(openai_text_request):
    """Test sync text generation returns a Response object"""
    client = OpenAIClientSync()
    result = client.query(openai_text_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message content
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0
    assert result.total_tokens > 0

    # Verify request is preserved
    assert result.request == openai_text_request


@pytest.mark.asyncio
async def test_openai_text_generation_async(openai_text_request):
    """Test async text generation returns a Response object"""
    client = OpenAIClientAsync()
    result = await client.query(openai_text_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message content
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0

    # Verify request is preserved
    assert result.request == openai_text_request


def test_openai_streaming_sync(openai_stream_request):
    """Test sync streaming returns a SyncStream object"""
    client = OpenAIClientSync()
    result = client.query(openai_stream_request)

    # For streaming, we should get a Stream object directly
    assert isinstance(result, SyncStream)


@pytest.mark.asyncio
async def test_openai_streaming_async(openai_stream_request):
    """Test async streaming returns an AsyncStream object"""
    client = OpenAIClientAsync()
    result = await client.query(openai_stream_request)

    # For streaming, we should get an AsyncStream object directly
    assert isinstance(result, AsyncStream)


def test_openai_structured_response_sync(openai_structured_request):
    """Test sync structured response returns a Response with structured content"""
    client = OpenAIClientSync()
    result = client.query(openai_structured_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify the parsed field contains the structured Frog object
    assert result.message.parsed is not None
    assert isinstance(result.message.parsed, Frog)
    assert hasattr(result.message.parsed, "species")
    assert hasattr(result.message.parsed, "age")
    assert hasattr(result.message.parsed, "name")
    assert hasattr(result.message.parsed, "occupation")
    assert hasattr(result.message.parsed, "color")
    assert hasattr(result.message.parsed, "legs")
    assert hasattr(result.message.parsed, "continent")

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_structured_response_async(openai_structured_request):
    """Test async structured response returns a Response with structured content"""
    client = OpenAIClientAsync()
    result = await client.query(openai_structured_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify the parsed field contains the structured Frog object
    assert result.message.parsed is not None
    assert isinstance(result.message.parsed, Frog)
    assert hasattr(result.message.parsed, "species")
    assert hasattr(result.message.parsed, "age")
    assert hasattr(result.message.parsed, "name")
    assert hasattr(result.message.parsed, "occupation")
    assert hasattr(result.message.parsed, "color")
    assert hasattr(result.message.parsed, "legs")
    assert hasattr(result.message.parsed, "continent")

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0


####################
# OpenAI Audio Tests
####################


def test_openai_audio_input_sync(openai_audio_input_request):
    """Test sync audio input (chat completions with audio)"""
    client = OpenAIClientSync()
    result = client.query(openai_audio_input_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message content (should transcribe or understand the audio)
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0

    # Verify request is preserved
    assert result.request == openai_audio_input_request


def test_openai_audio_generation_sync(openai_audio_generation_request):
    """Test sync audio generation (TTS)"""
    client = OpenAIClientSync()
    result = client.query(openai_audio_generation_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message contains audio data (base64 encoded)
    assert result.message is not None
    assert result.message.content is not None
    assert isinstance(result.message.content, str)
    assert len(result.message.content) > 0

    # The content should be base64-encoded audio
    # Basic check: base64 strings are typically much longer than plain text
    assert len(result.message.content) > 1000

    # Verify request is preserved
    assert result.request == openai_audio_generation_request


def test_openai_audio_transcription_sync(openai_audio_transcription_request):
    """Test sync audio transcription (Whisper)"""
    client = OpenAIClientSync()
    result = client.query(openai_audio_transcription_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message contains transcribed text
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Harvard sentences should contain recognizable words
    assert any(
        word.lower() in result.content.lower() for word in ["the", "a", "and", "of"]
    )

    # Verify request is preserved
    assert result.request == openai_audio_transcription_request


####################
# OpenAI Image Tests
####################


def test_openai_image_analysis_sync(openai_image_analysis_request):
    """Test sync image analysis (vision) using gpt-4o"""
    client = OpenAIClientSync()
    result = client.query(openai_image_analysis_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message content (should describe the image)
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0

    # Verify request is preserved
    assert result.request == openai_image_analysis_request


def test_openai_image_generation_sync(openai_image_generation_request):
    """Test sync image generation (DALL-E)"""
    client = OpenAIClientSync()
    result = client.query(openai_image_generation_request)

    # Verify it returns a Response object
    assert isinstance(result, Response)

    # Verify Response structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message contains image data in the images field
    assert result.message is not None
    assert result.message.images is not None
    assert len(result.message.images) > 0

    # Check the first image
    first_image = result.message.images[0]
    assert first_image.b64_json is not None
    assert len(first_image.b64_json) > 1000  # Should be a long base64 string

    # Verify request is preserved
    assert result.request == openai_image_generation_request
