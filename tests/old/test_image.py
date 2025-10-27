"""
Regression testing for ImageMessage functionality.
Core APIs.
"""
from Chain import Model, Response, Chain
from Chain.message.imagemessage import ImageMessage
from pytest import fixture
from pathlib import Path

# Get the fixtures directory
fixtures_dir = Path(__file__).parent.parent / "fixtures"
sample_image_file = fixtures_dir / "image.png"

@fixture
def vision_models() -> list[str]:
    """Models that support vision/image input"""
    return [
        "gpt-4o",      # OpenAI vision
        "claude",      # Anthropic vision
        "gemini2.5"    # Google vision
    ]

@fixture
def sample_image_message() -> ImageMessage:
    """Create a sample image message for testing"""
    return ImageMessage.from_image_file(
        role="user",
        text_content="What do you see in this image?",
        image_file=sample_image_file
    )

# ImageMessage creation
# --------------------------------------

def test_imagemessage_creation(sample_image_message):
    """Test basic ImageMessage object creation"""
    img_msg = sample_image_message
    assert img_msg.role == "user"
    assert img_msg.text_content == "What do you see in this image?"
    assert len(img_msg.image_content) > 0  # Has base64 content
    assert img_msg.mime_type == "image/png"
    assert img_msg.message_type == "image"

def test_imagemessage_from_file():
    """Test ImageMessage creation from file"""
    img_msg = ImageMessage.from_image_file(
        role="user",
        text_content="Describe this image",
        image_file=sample_image_file
    )
    assert isinstance(img_msg, ImageMessage)
    assert img_msg.role == "user"
    assert img_msg.text_content == "Describe this image"

# Model queries with ImageMessage
# --------------------------------------

def test_model_query_with_imagemessage():
    """Test single model query with ImageMessage"""
    model = Model("gpt-4o")
    img_msg = ImageMessage.from_image_file(
        role="user",
        text_content="What is in this image?",
        image_file=sample_image_file
    )
    
    response = model.query(img_msg)
    assert isinstance(response, Response)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

def test_chain_with_imagemessage():
    """Test Chain with ImageMessage"""
    model = Model("claude")
    chain = Chain(model=model)
    
    img_msg = ImageMessage.from_image_file(
        role="user",
        text_content="Please describe what you see",
        image_file=sample_image_file
    )
    
    response = chain.run(messages=[img_msg])
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

# Format conversion
# --------------------------------------

def test_imagemessage_to_openai():
    """Test ImageMessage OpenAI format conversion"""
    img_msg = ImageMessage.from_image_file(
        role="user",
        text_content="Test image",
        image_file=sample_image_file
    )
    
    openai_msg = img_msg.to_openai()
    assert openai_msg["role"] == "user"
    assert len(openai_msg["content"]) == 2  # Text + image content

def test_imagemessage_to_anthropic():
    """Test ImageMessage Anthropic format conversion"""
    img_msg = ImageMessage.from_image_file(
        role="user", 
        text_content="Test image",
        image_file=sample_image_file
    )
    
    anthropic_msg = img_msg.to_anthropic()
    assert anthropic_msg["role"] == "user"
    assert len(anthropic_msg["content"]) == 2  # Image + text content
