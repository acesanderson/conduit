"""
Regression testing for Chain caching functionality.
Tests caching across different interfaces and message types.
"""
from Chain import Model, ModelAsync, Chain, AsyncChain, Prompt, Parser, Response
from Chain.message.audiomessage import AudioMessage
from Chain.message.imagemessage import ImageMessage
from Chain.cache.cache import ChainCache
from Chain.tests.fixtures.sample_models import PydanticTestFrog
from pytest import fixture
from pathlib import Path
import tempfile
import time

# Get the fixtures directory
fixtures_dir = Path(__file__).parent.parent / "fixtures"
sample_audio_file = fixtures_dir / "audio.mp3"
sample_image_file = fixtures_dir / "image.png"

@fixture
def cache_db():
    """Create a temporary cache database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_path = f.name
    
    cache = ChainCache(cache_path)
    yield cache
    
    # Cleanup
    cache.close()
    Path(cache_path).unlink(missing_ok=True)

@fixture
def sample_audio_message():
    """Create a sample audio message for testing"""
    return AudioMessage.from_audio_file(
        role="user",
        text_content="Please transcribe this audio file.",
        audio_file=sample_audio_file
    )

@fixture
def sample_image_message():
    """Create a sample image message for testing"""
    return ImageMessage.from_image_file(
        role="user",
        text_content="What do you see in this image?",
        image_file=sample_image_file
    )

# Model-level caching tests
# --------------------------------------

def test_model_cache_text_query(cache_db):
    """Test Model caching with simple text query"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    query = "What is 2+2?"
    
    # First query - should hit API
    response1 = model.query(query, cache=True)
    assert isinstance(response1, Response)
    
    # Second query - should hit cache
    start_time = time.time()
    response2 = model.query(query, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert duration < 0.1  # Cache hit should be very fast
    assert str(response1.content) == str(response2.content)

def test_model_cache_disabled(cache_db):
    """Test Model with caching disabled"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    query = "What is 3+3?"
    
    # Query with cache disabled
    response1 = model.query(query, cache=False)
    response2 = model.query(query, cache=False)
    
    assert isinstance(response1, Response)
    assert isinstance(response2, Response)
    # Both should be API calls, content may vary

def test_model_cache_with_parser(cache_db):
    """Test Model caching with structured output"""
    Model._chain_cache = cache_db
    
    model = Model("gpt")
    parser = Parser(PydanticTestFrog)
    query = "Create a frog"
    
    # First query
    response1 = model.query(query, response_model=PydanticTestFrog, cache=True)
    assert isinstance(response1, Response)
    assert isinstance(response1.content, PydanticTestFrog)
    
    # Second query - should hit cache
    start_time = time.time()
    response2 = model.query(query, response_model=PydanticTestFrog, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert isinstance(response2.content, PydanticTestFrog)
    assert duration < 0.1  # Cache hit

def test_model_cache_audio_message(cache_db, sample_audio_message):
    """Test Model caching with AudioMessage"""
    Model._chain_cache = cache_db
    
    model = Model("gpt-4o-audio-preview")
    # model = Model("gemini")
    
    # First query
    response1 = model.query(query_input = [sample_audio_message], cache=True)
    assert isinstance(response1, Response)
    
    # Second query - should hit cache
    start_time = time.time()
    response2 = model.query(query_input = [sample_audio_message], cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert duration < 0.1  # Cache hit
    assert str(response1.content) == str(response2.content)

def test_model_cache_image_message(cache_db, sample_image_message):
    """Test Model caching with ImageMessage"""
    Model._chain_cache = cache_db
    
    model = Model("gpt-4o")
    
    # First query
    response1 = model.query(sample_image_message, cache=True)
    assert isinstance(response1, Response)
    
    # Second query - should hit cache
    start_time = time.time()
    response2 = model.query(sample_image_message, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert duration < 0.1  # Cache hit
    assert str(response1.content) == str(response2.content)

# Chain-level caching tests
# --------------------------------------

def test_chain_cache_text_completion(cache_db):
    """Test Chain caching with text completion"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    prompt = Prompt("What is the capital of France?")
    chain = Chain(model=model, prompt=prompt)
    
    # First run
    response1 = chain.run(cache=True)
    assert isinstance(response1, Response)
    
    # Second run - should hit cache
    start_time = time.time()
    response2 = chain.run(cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert duration < 0.1  # Cache hit
    assert str(response1.content) == str(response2.content)

def test_chain_cache_with_variables(cache_db):
    """Test Chain caching with input variables"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    prompt = Prompt("What is the capital of {{country}}?")
    chain = Chain(model=model, prompt=prompt)
    
    input_vars = {"country": "Germany"}
    
    # First run
    response1 = chain.run(input_variables=input_vars, cache=True)
    assert isinstance(response1, Response)
    
    # Second run with same variables - should hit cache
    start_time = time.time()
    response2 = chain.run(input_variables=input_vars, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert duration < 0.1  # Cache hit
    assert str(response1.content) == str(response2.content)

def test_chain_cache_with_parser(cache_db):
    """Test Chain caching with structured output"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    prompt = Prompt("Create a frog")
    parser = Parser(PydanticTestFrog)
    chain = Chain(model=model, prompt=prompt, parser=parser)
    
    # First run
    response1 = chain.run(cache=True)
    assert isinstance(response1, Response)
    assert isinstance(response1.content, PydanticTestFrog)
    
    # Second run - should hit cache
    start_time = time.time()
    response2 = chain.run(cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert isinstance(response2.content, PydanticTestFrog)
    assert duration < 0.1  # Cache hit

def test_chain_cache_with_messages(cache_db, sample_image_message):
    """Test Chain caching with message list"""
    Model._chain_cache = cache_db
    
    model = Model("gpt-4o")
    chain = Chain(model=model)
    
    messages = [sample_image_message]
    
    # First run
    response1 = chain.run(messages=messages, cache=True)
    assert isinstance(response1, Response)
    
    # Second run - should hit cache
    start_time = time.time()
    response2 = chain.run(messages=messages, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(response2, Response)
    assert duration < 0.1  # Cache hit

# Async caching tests
# --------------------------------------

def test_modelasync_cache_text_query(cache_db):
    """Test ModelAsync caching with simple text query"""
    ModelAsync._chain_cache = cache_db
    
    model = ModelAsync("gpt3")
    chain = AsyncChain(model=model)
    
    prompt_strings = ["What is 4+4?"]
    
    # First run
    responses1 = chain.run(prompt_strings=prompt_strings, cache=True)
    assert isinstance(responses1, list)
    assert len(responses1) == 1
    assert isinstance(responses1[0], Response)
    
    # Second run - should hit cache
    start_time = time.time()
    responses2 = chain.run(prompt_strings=prompt_strings, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(responses2, list)
    assert len(responses2) == 1
    assert isinstance(responses2[0], Response)
    assert duration < 0.5  # Cache hit should be faster
    assert str(responses1[0].content) == str(responses2[0].content)

def test_asyncchain_cache_with_variables(cache_db):
    """Test AsyncChain caching with input variables"""
    ModelAsync._chain_cache = cache_db
    
    model = ModelAsync("gpt3")
    prompt = Prompt("What is the capital of {{country}}?")
    chain = AsyncChain(model=model, prompt=prompt)
    
    input_variables_list = [{"country": "Italy"}]
    
    # First run
    responses1 = chain.run(input_variables_list=input_variables_list, cache=True)
    assert isinstance(responses1, list)
    assert len(responses1) == 1
    assert isinstance(responses1[0], Response)
    
    # Second run - should hit cache
    start_time = time.time()
    responses2 = chain.run(input_variables_list=input_variables_list, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(responses2, list)
    assert len(responses2) == 1
    assert isinstance(responses2[0], Response)
    assert duration < 0.5  # Cache hit should be faster
    assert str(responses1[0].content) == str(responses2[0].content)

def test_asyncchain_cache_with_parser(cache_db):
    """Test AsyncChain caching with structured output"""
    ModelAsync._chain_cache = cache_db
    
    model = ModelAsync("gpt3")
    parser = Parser(PydanticTestFrog)
    chain = AsyncChain(model=model, parser=parser)
    
    prompt_strings = ["Create a green frog"]
    
    # First run
    responses1 = chain.run(prompt_strings=prompt_strings, cache=True)
    assert isinstance(responses1, list)
    assert len(responses1) == 1
    assert isinstance(responses1[0], Response)
    assert isinstance(responses1[0].content, PydanticTestFrog)
    
    # Second run - should hit cache
    start_time = time.time()
    responses2 = chain.run(prompt_strings=prompt_strings, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(responses2, list)
    assert len(responses2) == 1
    assert isinstance(responses2[0], Response)
    assert isinstance(responses2[0].content, PydanticTestFrog)
    assert duration < 0.5  # Cache hit should be faster

def test_asyncchain_cache_multiple_queries(cache_db):
    """Test AsyncChain caching with multiple queries"""
    ModelAsync._chain_cache = cache_db
    
    model = ModelAsync("gpt3")
    chain = AsyncChain(model=model)
    
    prompt_strings = ["What is 5+5?", "What is 6+6?"]
    
    # First run
    responses1 = chain.run(prompt_strings=prompt_strings, cache=True)
    assert isinstance(responses1, list)
    assert len(responses1) == 2
    
    # Second run - should hit cache for both
    start_time = time.time()
    responses2 = chain.run(prompt_strings=prompt_strings, cache=True)
    duration = time.time() - start_time
    
    assert isinstance(responses2, list)
    assert len(responses2) == 2
    assert duration < 1.0  # Cache hits should be faster
    
    # Content should match
    for r1, r2 in zip(responses1, responses2):
        assert str(r1.content) == str(r2.content)

# Cache invalidation tests
# --------------------------------------

def test_cache_different_queries(cache_db):
    """Test that different queries don't hit same cache"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    
    # Two different queries
    response1 = model.query("What is 7+7?", cache=True)
    response2 = model.query("What is 8+8?", cache=True)
    
    assert isinstance(response1, Response)
    assert isinstance(response2, Response)
    # Different queries should have different results
    assert str(response1.content) != str(response2.content)

def test_cache_different_temperatures(cache_db):
    """Test that different temperatures create different cache entries"""
    Model._chain_cache = cache_db
    
    model = Model("gpt3")
    query = "Tell me a joke"
    
    # Same query, different temperatures
    response1 = model.query(query, temperature=0.1, cache=True)
    response2 = model.query(query, temperature=0.9, cache=True)
    
    assert isinstance(response1, Response)
    assert isinstance(response2, Response)
    # Different temperatures should potentially have different results
