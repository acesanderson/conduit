"""
Regression testing for AudioMessage functionality.
Core APIs.
"""
from Chain import Model, Response, Chain
from Chain.message.audiomessage import AudioMessage
from pytest import fixture
from pathlib import Path

# Get the fixtures directory
fixtures_dir = Path(__file__).parent.parent / "fixtures"
sample_audio_file = fixtures_dir / "audio.mp3"

@fixture
def audio_models() -> list[str]:
    """Models that support audio input"""
    return [
        "gpt-4o-audio-preview",  # OpenAI audio model
        "gemini2.5"              # Gemini audio support
    ]

@fixture
def sample_audio_message() -> AudioMessage:
    """Create a sample audio message for testing"""
    return AudioMessage.from_audio_file(
        role="user",
        text_content="Please transcribe this audio file.",
        audio_file=sample_audio_file
    )

# AudioMessage creation
# --------------------------------------

def test_audiomessage_creation(sample_audio_message):
    """Test basic AudioMessage object creation"""
    audio_msg = sample_audio_message
    assert audio_msg.role == "user"
    assert audio_msg.text_content == "Please transcribe this audio file."
    assert audio_msg.format in ["mp3", "wav"]
    assert len(audio_msg.audio_content) > 0  # Has base64 content
    assert audio_msg.message_type == "audio"

def test_audiomessage_from_file():
    """Test AudioMessage creation from file"""
    audio_msg = AudioMessage.from_audio_file(
        role="user", 
        text_content="Transcribe this",
        audio_file=sample_audio_file
    )
    assert isinstance(audio_msg, AudioMessage)
    assert audio_msg.role == "user"
    assert audio_msg.text_content == "Transcribe this"

# Model queries with AudioMessage
# --------------------------------------

def test_model_query_with_audiomessage():
    """Test single model query with AudioMessage"""
    model = Model("gpt-4o-audio-preview")
    audio_msg = AudioMessage.from_audio_file(
        role="user",
        text_content="Please transcribe this audio.",
        audio_file=sample_audio_file
    )
    
    response = model.query(audio_msg)
    assert isinstance(response, Response)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

def test_chain_with_audiomessage():
    """Test Chain with AudioMessage"""
    model = Model("gemini2.5")
    chain = Chain(model=model)
    
    audio_msg = AudioMessage.from_audio_file(
        role="user",
        text_content="What is said in this audio?",
        audio_file=sample_audio_file
    )
    
    response = chain.run(messages=[audio_msg])
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

# Format conversion
# --------------------------------------

def test_audiomessage_to_openai():
    """Test AudioMessage OpenAI format conversion"""
    audio_msg = AudioMessage.from_audio_file(
        role="user",
        text_content="Test transcription",
        audio_file=sample_audio_file
    )
    
    openai_msg = audio_msg.to_openai()
    assert openai_msg["role"] == "user"
    assert len(openai_msg["content"]) == 2  # Text + audio content
