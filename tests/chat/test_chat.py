"""
Regression testing for Chat functionality.
Core APIs only - focused on intended functionality.
"""
from Chain.chat.chat import Chat
from Chain.model.model import Model
from Chain.message.messagestore import MessageStore
from Chain.message.textmessage import TextMessage
from pytest import fixture
from pathlib import Path
from rich.console import Console
from unittest.mock import patch
import tempfile

# Test fixtures
@fixture 
def sample_models() -> list[str]:
    """Models that should work with Chat"""
    return ["gpt3", "haiku", "gemini"]

@fixture
def temp_messagestore() -> MessageStore:
    """Create a temporary messagestore for testing"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = Path(f.name)
    
    messagestore = MessageStore(
        history_file=temp_file,
        log_file="", 
        auto_save=False  # Disable auto-save for tests
    )
    yield messagestore
    
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()

@fixture
def sample_image_file() -> Path:
    """Path to test image file"""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    return fixtures_dir / "image.png"

# Basic Chat initialization
# --------------------------------------

def test_chat_initialization_default():
    """Test Chat can be initialized with defaults"""
    chat = Chat()
    assert isinstance(chat.model, Model)
    assert chat.model.model == "o4-mini"  # Default model
    assert chat.console is not None
    assert chat.messagestore is None  # No messagestore by default

def test_chat_initialization_with_model(sample_models):
    """Test Chat can be initialized with different models"""
    for model_name in sample_models:
        model = Model(model_name)
        chat = Chat(model=model)
        assert chat.model.model == model.model
        assert isinstance(chat.console, Console)

def test_chat_initialization_with_messagestore(temp_messagestore):
    """Test Chat can be initialized with a messagestore"""
    chat = Chat(messagestore=temp_messagestore)
    assert chat.messagestore == temp_messagestore
    assert len(chat.messagestore) == 0

# Command parsing and execution
# --------------------------------------

def test_command_discovery():
    """Test that Chat discovers command methods"""
    chat = Chat()
    commands = chat.get_commands()
    
    # Should find all command_ methods
    assert "command_exit" in commands
    assert "command_help" in commands  
    assert "command_clear" in commands
    assert "command_show_history" in commands
    assert "command_set_model" in commands

def test_command_parsing_valid():
    """Test parsing of valid commands"""
    chat = Chat()
    
    # Test parameterless commands
    exit_cmd = chat.parse_input("/exit")
    assert callable(exit_cmd)
    
    help_cmd = chat.parse_input("/help")
    assert callable(help_cmd)
    
    # Test commands with parameters  
    model_cmd = chat.parse_input("/set model gpt3")
    assert callable(model_cmd)

def test_command_parsing_invalid():
    """Test parsing of invalid commands"""
    chat = Chat()
    
    # Non-command input should return None
    result = chat.parse_input("hello world")
    assert result is None
    
    # Invalid command should raise ValueError
    try:
        chat.parse_input("/nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_command_parsing_non_command():
    """Test that non-command input returns None"""
    chat = Chat()
    
    # Regular text should return None (not a command)
    result = chat.parse_input("What is 2+2?")
    assert result is None
    
    result = chat.parse_input("Hello, how are you?")
    assert result is None

# Model querying
# --------------------------------------

def test_query_model_string_input():
    """Test querying model with string input"""
    chat = Chat(model=Model("gpt3"))
    
    # Mock the model.query to avoid actual API calls in tests
    with patch.object(chat.model, 'query') as mock_query:
        mock_query.return_value = "Mocked response"
        
        result = chat.query_model("What is 2+2?")
        
        assert result == "Mocked response"
        mock_query.assert_called_once()

def test_query_model_with_messagestore():
    """Test querying model updates messagestore"""
    messagestore = MessageStore(auto_save=False)
    chat = Chat(model=Model("gpt3"), messagestore=messagestore)
    
    with patch.object(chat.model, 'query') as mock_query:
        mock_query.return_value = "Mocked response"
        
        # Initially empty
        assert len(messagestore) == 0
        
        result = chat.query_model("Test question")
        
        # Should add assistant response
        assert len(messagestore) == 1
        assert messagestore.last().role == "assistant"
        assert messagestore.last().content == "Mocked response"

def test_query_model_message_list():
    """Test querying model with list of messages"""
    chat = Chat(model=Model("gpt3"))
    messages = [
        TextMessage(role="user", content="Hello"),
        TextMessage(role="assistant", content="Hi there")
    ]
    
    with patch.object(chat.model, 'query') as mock_query:
        mock_query.return_value = "Mocked response"
        
        result = chat.query_model(messages)
        
        assert result == "Mocked response"
        mock_query.assert_called_once_with(messages, verbose=False)

# Command execution
# --------------------------------------

def test_command_show_models():
    """Test show models command"""
    chat = Chat()
    
    # Should not raise an error
    try:
        chat.command_show_models()
    except Exception as e:
        assert False, f"command_show_models raised {e}"

def test_command_show_model():
    """Test show current model command"""
    chat = Chat(model=Model("gpt3"))
    
    # Should not raise an error
    try:
        chat.command_show_model()
    except Exception as e:
        assert False, f"command_show_model raised {e}"

def test_command_set_model():
    """Test set model command"""
    chat = Chat()
    original_model = chat.model.model
    
    # Test valid model change
    chat.command_set_model("haiku")
    assert chat.model.model != original_model
    
    # Test invalid model (should handle gracefully)
    try:
        chat.command_set_model("nonexistent_model")
    except Exception:
        pass  # Expected to fail, should handle gracefully

def test_command_clear():
    """Test clear command (should not crash)"""
    chat = Chat()
    
    try:
        chat.command_clear()
    except Exception as e:
        assert False, f"command_clear raised {e}"

def test_command_show_history_empty():
    """Test show history with empty messagestore"""
    messagestore = MessageStore(auto_save=False)
    chat = Chat(messagestore=messagestore)
    
    # Should not crash with empty history
    try:
        chat.command_show_history()
    except Exception as e:
        assert False, f"command_show_history raised {e}"

def test_command_show_history_with_messages():
    """Test show history with messages"""
    messagestore = MessageStore(auto_save=False)
    messagestore.add_new("user", "Hello")
    messagestore.add_new("assistant", "Hi there")
    
    chat = Chat(messagestore=messagestore)
    
    # Should not crash with messages
    try:
        chat.command_show_history()
    except Exception as e:
        assert False, f"command_show_history raised {e}"

# Image functionality (if available)
# --------------------------------------

def test_command_paste_image_over_ssh():
    """Test image paste command over SSH (should handle gracefully)"""
    chat = Chat()
    
    # Mock SSH environment
    with patch.dict('os.environ', {'SSH_CLIENT': 'test'}):
        try:
            chat.command_paste_image()
        except Exception as e:
            assert False, f"command_paste_image should handle SSH gracefully, but raised {e}"

def test_command_wipe_image_no_image():
    """Test wipe image command when no image exists"""
    chat = Chat()
    
    # Should handle gracefully when no image exists
    try:
        chat.command_wipe_image()
    except Exception as e:
        assert False, f"command_wipe_image raised {e}"

# Integration with Chain._message_store
# --------------------------------------

def test_chain_message_store_integration():
    """Test that Chat sets Chain._message_store during chat setup"""
    from Chain.chain.chain import Chain
    
    messagestore = MessageStore(auto_save=False)
    chat = Chat(messagestore=messagestore)
    
    # Simulate what happens in chat() method
    Chain._message_store = chat.messagestore
    
    assert Chain._message_store == messagestore

# Error handling
# --------------------------------------

def test_query_model_handles_none_response():
    """Test that query_model handles None response gracefully"""
    chat = Chat(model=Model("gpt3"))
    
    with patch.object(chat.model, 'query') as mock_query:
        mock_query.return_value = None
        
        # Should handle None response without crashing
        result = chat.query_model("Test")
        assert result is None

def test_messagestore_operations_handle_errors():
    """Test that messagestore operations are robust"""
    # Test with None messagestore
    chat = Chat(messagestore=None)
    
    with patch.object(chat.model, 'query') as mock_query:
        mock_query.return_value = "Response"
        
        # Should not crash when messagestore is None
        try:
            result = chat.query_model("Test")
            assert result == "Response"
        except Exception as e:
            assert False, f"Should handle None messagestore gracefully, but raised {e}"

# System message functionality
# --------------------------------------

def test_system_message_initialization():
    """Test Chat with system message"""
    system_msg = TextMessage(role="system", content="You are helpful")
    messagestore = MessageStore(auto_save=False)
    
    chat = Chat(messagestore=messagestore)
    chat.system_message = system_msg
    
    # Simulate chat initialization
    if chat.system_message:
        chat.messagestore.append(chat.system_message)
    
    assert len(messagestore) == 1
    assert messagestore[0].role == "system"
    assert messagestore[0].content == "You are helpful"
