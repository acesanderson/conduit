"""
Regression testing for MessageStore functionality.
Core APIs only - focused on intended functionality.
"""
from Chain import Chain, Model, Response
from Chain.message.messagestore import MessageStore
from Chain.message.textmessage import TextMessage
from pytest import fixture
from pathlib import Path
import tempfile

# Test fixtures
@fixture
def temp_messagestore() -> MessageStore:
    """Create a temporary messagestore for testing"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = Path(f.name)
    
    messagestore = MessageStore(
        history_file=temp_file,
        auto_save=False  # Disable auto-save for tests
    )
    yield messagestore
    
    # Cleanup
    if temp_file.exists():
        temp_file.unlink()

@fixture
def sample_models() -> list[str]:
    """Models for testing"""
    return ["gpt3", "haiku", "gemini"]

# Basic MessageStore initialization and behavior
# --------------------------------------

def test_messagestore_initialization():
    """Test MessageStore can be initialized"""
    messagestore = MessageStore()
    assert len(messagestore) == 0
    assert not messagestore.persistent
    assert not messagestore.logging
    assert messagestore.auto_save == True

def test_messagestore_initialization_with_persistence(temp_messagestore):
    """Test MessageStore with persistence"""
    messagestore = temp_messagestore
    assert messagestore.persistent
    assert messagestore.history_file.exists()
    assert messagestore.db is not None

def test_messagestore_behaves_like_list():
    """Test MessageStore list-like behavior"""
    messagestore = MessageStore()
    
    # Test append
    msg1 = TextMessage(role="user", content="Hello")
    messagestore.append(msg1)
    assert len(messagestore) == 1
    
    # Test indexing
    assert messagestore[0] == msg1
    assert messagestore.last() == msg1
    
    # Test add_new convenience method
    messagestore.add_new("assistant", "Hi there")
    assert len(messagestore) == 2
    assert messagestore.last().role == "assistant"
    assert messagestore.last().content == "Hi there"

def test_messagestore_clear_and_view():
    """Test clear and view history methods"""
    messagestore = MessageStore()
    
    # Add some messages
    messagestore.add_new("user", "Hello")
    messagestore.add_new("assistant", "Hi")
    assert len(messagestore) == 2
    
    # Test view_history (should not crash)
    messagestore.view_history()
    
    # Test clear
    messagestore.clear()
    assert len(messagestore) == 0

# Persistence functionality
# --------------------------------------

def test_messagestore_save_and_load(temp_messagestore):
    """Test save and load functionality"""
    messagestore = temp_messagestore
    
    # Add messages
    messagestore.add_new("user", "Test message 1")
    messagestore.add_new("assistant", "Test response 1")
    messagestore.add_new("user", "Test message 2")
    
    # Verify messages are there
    assert len(messagestore) == 3
    
    # Save manually
    messagestore.save()
    
    # Verify database file exists and has content
    assert messagestore.history_file.exists()
    assert messagestore.db is not None
    
    # Create new messagestore with same file
    new_messagestore = MessageStore(history_file=messagestore.history_file, auto_save=False)
    
    # Should start empty
    assert len(new_messagestore) == 0
    
    # Load from database
    new_messagestore.load()
    
    # Check messages were restored
    assert len(new_messagestore) == 3
    assert new_messagestore[0].role == "user"
    assert new_messagestore[0].content == "Test message 1"
    assert new_messagestore[2].content == "Test message 2"

def test_messagestore_auto_save():
    """Test auto-save functionality"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = Path(f.name)
    
    try:
        # Create messagestore with auto-save enabled
        messagestore = MessageStore(history_file=temp_file, auto_save=True)
        
        # Add a message (should auto-save)
        messagestore.add_new("user", "Auto-save test")
        
        # Create new messagestore and load
        new_messagestore = MessageStore(history_file=temp_file)
        new_messagestore.load()
        
        # Message should be there
        assert len(new_messagestore) == 1
        assert new_messagestore[0].content == "Auto-save test"
        
    finally:
        if temp_file.exists():
            temp_file.unlink()

# Chain integration
# --------------------------------------

def test_chain_messagestore_integration():
    """Test Chain class using MessageStore"""
    messagestore = MessageStore()
    
    # Set Chain to use our messagestore
    Chain.message_store = messagestore
    
    model = Model("gpt3")
    chain = Chain(model=model)
    
    # Initially empty
    assert len(messagestore) == 0
    
    # Run a query (this should update the messagestore)
    response = chain.run(input_variables={}, messages=[
        TextMessage(role="user", content="What is 2+2?")
    ])
    
    # Check response is valid
    assert isinstance(response, Response)
    
    # Check messagestore was updated
    assert len(messagestore) >= 1  # At least the response should be there
    
    # Clean up
    Chain.message_store = None

def test_messagestore_add_response():
    """Test add_response method with Chain Response objects"""
    messagestore = MessageStore()
    model = Model("gpt3")
    chain = Chain(model=model)
    
    # Create a response by running the chain
    user_message = TextMessage(role="user", content="Hello")
    response = chain.run(messages=[user_message])
    
    # Should be able to add the response
    messagestore.add_response(response)
    
    # Should have both user and assistant messages
    assert len(messagestore) == 2
    assert messagestore[0].role == "user"
    assert messagestore[0].content == "Hello"
    assert messagestore[1].role == "assistant"
    assert isinstance(messagestore[1].content, str)

def test_messagestore_query_failed():
    """Test query_failed method"""
    messagestore = MessageStore()
    
    # Add user message
    messagestore.add_new("user", "This query will fail")
    assert len(messagestore) == 1
    
    # Simulate query failure
    messagestore.query_failed()
    
    # User message should be removed
    assert len(messagestore) == 0

# Error handling and edge cases
# --------------------------------------

def test_messagestore_handles_empty_state():
    """Test MessageStore handles empty state gracefully"""
    messagestore = MessageStore()
    
    # Should handle empty operations gracefully
    assert messagestore.last() is None
    assert messagestore.get(1) is None
    
    # View history should not crash on empty store
    messagestore.view_history()
    
    # Clear should not crash on empty store
    messagestore.clear()

def test_messagestore_copy():
    """Test MessageStore copy functionality"""
    messagestore = MessageStore()
    messagestore.add_new("user", "Original message")
    
    # Copy should work
    copied = messagestore.copy()
    assert len(copied) == 1
    assert copied[0].content == "Original message"
    
    # Copy should not be persistent (even if original was)
    assert not copied.persistent

def test_messagestore_get_by_role():
    """Test role-based message retrieval"""
    messagestore = MessageStore()
    
    messagestore.add_new("system", "You are helpful")
    messagestore.add_new("user", "Hello")
    messagestore.add_new("assistant", "Hi there")
    messagestore.add_new("user", "How are you?")
    
    # Test role filtering
    user_messages = messagestore.user_messages()
    assert len(user_messages) == 2
    assert all(msg.role == "user" for msg in user_messages)
    
    assistant_messages = messagestore.assistant_messages()
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "Hi there"
    
    system_messages = messagestore.system_messages()
    assert len(system_messages) == 1
    assert system_messages[0].content == "You are helpful"

def test_messagestore_pruning():
    """Test automatic pruning functionality"""
    messagestore = MessageStore(pruning=True, auto_save=False)
    
    # Add more than 20 messages
    for i in range(25):
        messagestore.add_new("user", f"Message {i}")
        messagestore.add_new("assistant", f"Response {i}")
    
    assert len(messagestore) == 50  # 25 pairs
    
    # Trigger pruning manually
    messagestore.prune()
    
    # Should be pruned to last 20 messages
    assert len(messagestore) == 20
    
    # Should have the most recent messages
    assert "Message 24" in messagestore.last().content
