from __future__ import annotations
from conduit.core.clients.openai.client import OpenAIClient
from conduit.core.clients.google.client import GoogleClient
from conduit.core.clients.ollama.client import OllamaClient
from conduit.core.clients.perplexity.client import PerplexityClient
from conduit.core.clients.anthropic.client import AnthropicClient
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.request.request import GenerationRequest
from conduit.domain.result.response import GenerationResponse
from conduit.domain.message.message import (
    UserMessage,
    SystemMessage,
    TextContent,
    AudioContent,
    ImageContent,
)
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.core.parser.stream.protocol import AsyncStream
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
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def openai_stream_request():
    """Streaming text generation request"""
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    params = GenerationParams(model=model, stream=True)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def openai_structured_request():
    """Structured response request"""
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    params = GenerationParams(model=model, response_model=Frog)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


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
    params = GenerationParams(model=model, output_type="text")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def openai_audio_generation_request():
    """Audio generation (TTS) request"""
    user_message = UserMessage(content="Hello, this is a test of text to speech.")
    messages = [user_message]

    # Default TTS model
    model = "tts-1"
    params = GenerationParams(model=model, output_type="audio")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


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
    params = GenerationParams(model=model, output_type="transcription")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


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
    params = GenerationParams(model=model, output_type="text")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def openai_image_generation_request():
    """Image generation (DALL-E) request"""
    user_message = UserMessage(
        content="A serene landscape with mountains and a lake at sunset"
    )
    messages = [user_message]

    # Default DALL-E model
    model = "dall-e-3"
    params = GenerationParams(model=model, output_type="image")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


####################
# OpenAI Tests
####################


@pytest.mark.asyncio
async def test_openai_text_generation(openai_text_request):
    """Test text generation returns a GenerationResponse object"""
    client = OpenAIClient()
    result = await client.query(openai_text_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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


@pytest.mark.asyncio
async def test_openai_streaming(openai_stream_request):
    """Test streaming returns an AsyncStream object"""
    client = OpenAIClient()
    result = await client.query(openai_stream_request)

    # For streaming, we should get an AsyncStream object directly
    assert isinstance(result, AsyncStream)


@pytest.mark.asyncio
async def test_openai_structured_response(openai_structured_request):
    """Test structured response returns a GenerationResponse with structured content"""
    client = OpenAIClient()
    result = await client.query(openai_structured_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

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


@pytest.mark.asyncio
async def test_openai_audio_input(openai_audio_input_request):
    """Test audio input (chat completions with audio)"""
    client = OpenAIClient()
    result = await client.query(openai_audio_input_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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


@pytest.mark.asyncio
async def test_openai_audio_generation(openai_audio_generation_request):
    """Test audio generation (TTS)"""
    client = OpenAIClient()
    result = await client.query(openai_audio_generation_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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


@pytest.mark.asyncio
async def test_openai_audio_transcription(openai_audio_transcription_request):
    """Test audio transcription (Whisper)"""
    client = OpenAIClient()
    result = await client.query(openai_audio_transcription_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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


@pytest.mark.asyncio
async def test_openai_image_analysis(openai_image_analysis_request):
    """Test image analysis (vision) using gpt-4o"""
    client = OpenAIClient()
    result = await client.query(openai_image_analysis_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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


@pytest.mark.asyncio
async def test_openai_image_generation(openai_image_generation_request):
    """Test image generation (DALL-E)"""
    client = OpenAIClient()
    result = await client.query(openai_image_generation_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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


####################
# Google Gemini Fixtures
####################


@pytest.fixture
def google_text_request():
    """Basic text generation request for Google Gemini"""
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "flash"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def google_stream_request():
    """Streaming text generation request for Google Gemini"""
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "flash"
    params = GenerationParams(model=model, stream=True)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def google_structured_request():
    """Structured response request for Google Gemini"""
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "flash"
    params = GenerationParams(model=model, response_model=Frog)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def google_image_analysis_request():
    """Image analysis (vision) request for Google Gemini"""
    # Load the image file
    image_url = load_image_as_base64(IMAGE_PATH)

    # Create multimodal content with text and image
    text_content = TextContent(text="What is in this image? Describe it in detail.")
    image_content = ImageContent(url=image_url, detail="high")

    user_message = UserMessage(content=[text_content, image_content])
    messages = [user_message]

    # Use the vision-capable model
    model = "flash"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def google_audio_input_request():
    """Audio input request for Google Gemini (audio recognition)"""
    # Load the audio file
    audio_base64 = load_audio_as_base64(AUDIO_PATH)

    # Create multimodal content with text and audio
    text_content = TextContent(text="What does this audio say? Transcribe it.")
    audio_content = AudioContent(data=audio_base64, format="mp3")

    user_message = UserMessage(content=[text_content, audio_content])
    messages = [user_message]

    # Use the flash model (supports multimodal)
    model = "flash"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


####################
# Google Gemini Tests
####################


@pytest.mark.asyncio
async def test_google_text_generation(google_text_request):
    """Test Google Gemini text generation returns a GenerationResponse object"""
    client = GoogleClient()
    result = await client.query(google_text_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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
    assert result.request == google_text_request


@pytest.mark.asyncio
async def test_google_streaming(google_stream_request):
    """Test Google Gemini streaming returns an AsyncStream object"""
    client = GoogleClient()
    result = await client.query(google_stream_request)

    # For streaming, we should get an AsyncStream object directly
    assert isinstance(result, AsyncStream)


@pytest.mark.asyncio
async def test_google_structured_response(google_structured_request):
    """Test Google Gemini structured response returns a GenerationResponse with structured content"""
    client = GoogleClient()
    result = await client.query(google_structured_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

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
async def test_google_image_analysis(google_image_analysis_request):
    """Test Google Gemini image analysis (vision) capability"""
    client = GoogleClient()
    result = await client.query(google_image_analysis_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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
    assert result.request == google_image_analysis_request


@pytest.mark.asyncio
async def test_google_audio_analysis(google_audio_input_request):
    """Test Google Gemini audio analysis (audio recognition) capability"""
    client = GoogleClient()
    result = await client.query(google_audio_input_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message content (should transcribe/describe the audio)
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0

    # Verify request is preserved
    assert result.request == google_audio_input_request


@pytest.mark.asyncio
async def test_google_metadata_structure(google_text_request):
    """Test that Google Gemini responses have proper metadata structure"""
    client = GoogleClient()
    result = await client.query(google_text_request)

    # Verify metadata structure
    assert hasattr(result.metadata, "duration")
    assert hasattr(result.metadata, "model_slug")
    assert hasattr(result.metadata, "input_tokens")
    assert hasattr(result.metadata, "output_tokens")
    assert hasattr(result.metadata, "stop_reason")
    assert hasattr(result.metadata, "timestamp")

    # Verify metadata types and values
    assert isinstance(result.metadata.duration, (int, float))
    assert result.metadata.duration > 0
    assert isinstance(result.metadata.model_slug, str)
    assert len(result.metadata.model_slug) > 0
    assert isinstance(result.metadata.input_tokens, int)
    assert isinstance(result.metadata.output_tokens, int)
    assert result.metadata.stop_reason is not None


@pytest.fixture
def google_image_generation_request():
    """Image generation request for Google Gemini (Imagen)"""
    user_message = UserMessage(content="A portrait of a sheepadoodle wearing a cape")
    messages = [user_message]
    model = "imagen-3.0-generate-002"
    params = GenerationParams(model=model, output_type="image")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def google_audio_generation_request():
    """Audio generation (TTS) request for Google Gemini"""
    user_message = UserMessage(content="Hello from Google Gemini text to speech.")
    messages = [user_message]
    model = "gemini-2.5-flash-preview-tts"
    params = GenerationParams(model=model, output_type="audio")
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.mark.asyncio
async def test_google_image_generation(google_image_generation_request):
    """Test Google Imagen image generation"""
    client = GoogleClient()
    result = await client.query(google_image_generation_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message contains image data
    assert result.message is not None
    assert result.message.images is not None
    assert isinstance(result.message.images, list)
    assert len(result.message.images) > 0

    # Verify the first image has either URL or b64_json
    first_image = result.message.images[0]
    assert hasattr(first_image, "url") or hasattr(first_image, "b64_json")
    assert first_image.url is not None or first_image.b64_json is not None

    # Verify request is preserved
    assert result.request == google_image_generation_request


@pytest.mark.asyncio
async def test_google_audio_generation(google_audio_generation_request):
    """Test Google Gemini audio generation (TTS)"""
    client = GoogleClient()
    result = await client.query(google_audio_generation_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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
    assert result.request == google_audio_generation_request


####################
# Ollama Fixtures
####################


@pytest.fixture
def ollama_text_request():
    """Basic text generation request for Ollama"""
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "llama3.1:latest"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def ollama_stream_request():
    """Streaming text generation request for Ollama"""
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "llama3.1:latest"
    params = GenerationParams(model=model, stream=True)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def ollama_structured_request():
    """Structured response request for Ollama"""
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "llama3.1:latest"
    params = GenerationParams(model=model, response_model=Frog)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


####################
# Ollama Tests
####################


@pytest.mark.asyncio
async def test_ollama_text_generation(ollama_text_request):
    """Test Ollama text generation returns a GenerationResponse object"""
    client = OllamaClient()
    result = await client.query(ollama_text_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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
    assert result.request == ollama_text_request


@pytest.mark.asyncio
async def test_ollama_streaming(ollama_stream_request):
    """Test Ollama streaming returns an AsyncStream object"""
    client = OllamaClient()
    result = await client.query(ollama_stream_request)

    # For streaming, we should get an AsyncStream object directly
    assert isinstance(result, AsyncStream)


@pytest.mark.asyncio
async def test_ollama_structured_response(ollama_structured_request):
    """Test Ollama structured response returns a GenerationResponse with structured content"""
    client = OllamaClient()
    result = await client.query(ollama_structured_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

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
async def test_ollama_metadata_structure(ollama_text_request):
    """Test that Ollama responses have proper metadata structure"""
    client = OllamaClient()
    result = await client.query(ollama_text_request)

    # Verify metadata structure
    assert hasattr(result.metadata, "duration")
    assert hasattr(result.metadata, "model_slug")
    assert hasattr(result.metadata, "input_tokens")
    assert hasattr(result.metadata, "output_tokens")
    assert hasattr(result.metadata, "stop_reason")
    assert hasattr(result.metadata, "timestamp")

    # Verify metadata types and values
    assert isinstance(result.metadata.duration, (int, float))
    assert result.metadata.duration > 0
    assert isinstance(result.metadata.model_slug, str)
    assert len(result.metadata.model_slug) > 0
    assert isinstance(result.metadata.input_tokens, int)
    assert isinstance(result.metadata.output_tokens, int)
    assert result.metadata.stop_reason is not None


####################
# Perplexity Fixtures
####################


@pytest.fixture
def perplexity_text_request():
    """Basic text generation request for Perplexity"""
    user_message = UserMessage(content="What is the capital of France?")
    messages = [user_message]
    model = "llama-3.1-sonar-small-128k-online"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def perplexity_stream_request():
    """Streaming text generation request for Perplexity"""
    user_message = UserMessage(
        content="What are the latest developments in quantum computing?"
    )
    messages = [user_message]
    model = "llama-3.1-sonar-small-128k-online"
    params = GenerationParams(model=model, stream=True)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def perplexity_structured_request():
    """Structured response request for Perplexity"""
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "llama-3.1-sonar-small-128k-online"
    params = GenerationParams(model=model, response_model=Frog)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


####################
# Perplexity Tests
####################


@pytest.mark.asyncio
async def test_perplexity_text_generation(perplexity_text_request):
    """Test Perplexity text generation returns a GenerationResponse object with citations"""
    client = PerplexityClient()
    result = await client.query(perplexity_text_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
    assert hasattr(result, "message")
    assert hasattr(result, "request")
    assert hasattr(result, "metadata")

    # Verify the message content is a dict (Perplexity format)
    assert result.message is not None
    assert isinstance(result.message.content, dict)
    assert "text" in result.message.content
    assert "citations" in result.message.content

    # Verify the text content
    assert isinstance(result.message.content["text"], str)
    assert len(result.message.content["text"]) > 0

    # Verify citations structure
    assert isinstance(result.message.content["citations"], list)

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0
    assert result.total_tokens > 0

    # Verify request is preserved
    assert result.request == perplexity_text_request


@pytest.mark.asyncio
async def test_perplexity_content_property(perplexity_text_request):
    """Test that perplexity_content property creates rich PerplexityContent object"""
    client = PerplexityClient()
    result = await client.query(perplexity_text_request)

    # Access the perplexity_content property
    perplexity_content = result.message.perplexity_content

    # Verify it's not None and has the right structure
    assert perplexity_content is not None
    assert hasattr(perplexity_content, "text")
    assert hasattr(perplexity_content, "citations")

    # Verify the text matches
    assert perplexity_content.text == result.message.content["text"]

    # Verify citations are PerplexityCitation objects
    assert isinstance(perplexity_content.citations, list)
    for citation in perplexity_content.citations:
        assert hasattr(citation, "title")
        assert hasattr(citation, "url")

    # Verify the __str__ method works (formats nicely)
    formatted = str(perplexity_content)
    assert result.message.content["text"] in formatted
    if perplexity_content.citations:
        assert "Sources:" in formatted


@pytest.mark.asyncio
async def test_perplexity_streaming(perplexity_stream_request):
    """Test Perplexity streaming returns an AsyncStream object"""
    client = PerplexityClient()
    result = await client.query(perplexity_stream_request)

    # For streaming, we should get an AsyncStream object directly
    assert isinstance(result, AsyncStream)


@pytest.mark.asyncio
async def test_perplexity_structured_response(perplexity_structured_request):
    """Test Perplexity structured response returns a GenerationResponse with structured content and citations"""
    client = PerplexityClient()
    result = await client.query(perplexity_structured_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify the parsed field contains the structured Frog object
    assert result.message.parsed is not None
    assert isinstance(result.message.parsed, Frog)
    assert hasattr(result.message.parsed, "species")
    assert hasattr(result.message.parsed, "age")
    assert hasattr(result.message.parsed, "name")

    # Verify content has citations (but no text for structured responses)
    assert isinstance(result.message.content, dict)
    assert "citations" in result.message.content
    assert "text" not in result.message.content

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0


@pytest.mark.asyncio
async def test_perplexity_structured_content_property(perplexity_structured_request):
    """Test that perplexity_content property works for structured responses"""
    client = PerplexityClient()
    result = await client.query(perplexity_structured_request)

    # Access the perplexity_content property
    perplexity_content = result.message.perplexity_content

    # Verify it's not None
    assert perplexity_content is not None

    # For structured responses, text should be JSON representation of parsed object
    assert perplexity_content.text is not None
    assert len(perplexity_content.text) > 0

    # Verify it contains the Frog data
    import json

    parsed_json = json.loads(perplexity_content.text)
    assert "species" in parsed_json
    assert "age" in parsed_json

    # Verify citations are still accessible
    assert hasattr(perplexity_content, "citations")


@pytest.mark.asyncio
async def test_perplexity_metadata_structure(perplexity_text_request):
    """Test that Perplexity responses have proper metadata structure"""
    client = PerplexityClient()
    result = await client.query(perplexity_text_request)

    # Verify metadata structure
    assert hasattr(result.metadata, "duration")
    assert hasattr(result.metadata, "model_slug")
    assert hasattr(result.metadata, "input_tokens")
    assert hasattr(result.metadata, "output_tokens")
    assert hasattr(result.metadata, "stop_reason")
    assert hasattr(result.metadata, "timestamp")

    # Verify metadata types and values
    assert isinstance(result.metadata.duration, (int, float))
    assert result.metadata.duration > 0
    assert isinstance(result.metadata.model_slug, str)
    assert len(result.metadata.model_slug) > 0
    assert isinstance(result.metadata.input_tokens, int)
    assert isinstance(result.metadata.output_tokens, int)
    assert result.metadata.stop_reason is not None


####################
# Anthropic Fixtures
####################


@pytest.fixture
def anthropic_text_request():
    """Basic text generation request for Anthropic"""
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "claude-3-5-sonnet-20241022"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def anthropic_system_message_request():
    """Request with system message for Anthropic (tests system message extraction)"""
    system_message = SystemMessage(content="You are a helpful assistant.")
    user_message = UserMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    model = "claude-3-5-sonnet-20241022"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def anthropic_stream_request():
    """Streaming text generation request for Anthropic"""
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "claude-3-5-sonnet-20241022"
    params = GenerationParams(model=model, stream=True)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def anthropic_structured_request():
    """Structured response request for Anthropic"""
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "claude-3-5-sonnet-20241022"
    params = GenerationParams(model=model, response_model=Frog)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


@pytest.fixture
def anthropic_image_analysis_request():
    """Image analysis request for Anthropic (vision capability)"""
    # Create a simple image message with a cat photo
    image_message = UserMessage(
        content=[
            TextContent(type="text", text="What's in this image?"),
            ImageContent(
                type="image_url",
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            ),
        ]
    )
    messages = [image_message]
    model = "claude-3-5-sonnet-20241022"
    params = GenerationParams(model=model)
    options = ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS)
    return GenerationRequest(params=params, messages=messages, options=options)


####################
# Anthropic Tests
####################


@pytest.mark.asyncio
async def test_anthropic_text_generation(anthropic_text_request):
    """Test Anthropic text generation returns a GenerationResponse object"""
    client = AnthropicClient()
    result = await client.query(anthropic_text_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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
    assert result.request == anthropic_text_request


@pytest.mark.asyncio
async def test_anthropic_system_message_handling(anthropic_system_message_request):
    """Test that Anthropic correctly extracts and handles system messages"""
    client = AnthropicClient()
    result = await client.query(anthropic_system_message_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify the message content
    assert result.message is not None
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify metadata has usage information
    assert result.metadata.input_tokens > 0
    assert result.metadata.output_tokens > 0

    # Verify request is preserved
    assert result.request == anthropic_system_message_request


@pytest.mark.asyncio
async def test_anthropic_streaming(anthropic_stream_request):
    """Test Anthropic streaming returns an AsyncStream object"""
    client = AnthropicClient()
    result = await client.query(anthropic_stream_request)

    # For streaming, we should get an AsyncStream object directly
    assert isinstance(result, AsyncStream)


@pytest.mark.asyncio
async def test_anthropic_structured_response(anthropic_structured_request):
    """Test Anthropic structured response returns a GenerationResponse with structured content"""
    client = AnthropicClient()
    result = await client.query(anthropic_structured_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

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
async def test_anthropic_image_analysis(anthropic_image_analysis_request):
    """Test Anthropic image analysis (vision) capability"""
    client = AnthropicClient()
    result = await client.query(anthropic_image_analysis_request)

    # Verify it returns a GenerationResponse object
    assert isinstance(result, GenerationResponse)

    # Verify GenerationResponse structure
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
    assert result.request == anthropic_image_analysis_request


@pytest.mark.asyncio
async def test_anthropic_metadata_structure(anthropic_text_request):
    """Test that Anthropic responses have proper metadata structure"""
    client = AnthropicClient()
    result = await client.query(anthropic_text_request)

    # Verify metadata structure
    assert hasattr(result.metadata, "duration")
    assert hasattr(result.metadata, "model_slug")
    assert hasattr(result.metadata, "input_tokens")
    assert hasattr(result.metadata, "output_tokens")
    assert hasattr(result.metadata, "stop_reason")
    assert hasattr(result.metadata, "timestamp")

    # Verify metadata types and values
    assert isinstance(result.metadata.duration, (int, float))
    assert result.metadata.duration > 0
    assert isinstance(result.metadata.model_slug, str)
    assert len(result.metadata.model_slug) > 0
    assert isinstance(result.metadata.input_tokens, int)
    assert isinstance(result.metadata.output_tokens, int)
    assert result.metadata.stop_reason is not None
