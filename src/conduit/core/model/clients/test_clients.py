from conduit.core.model.clients.openai.client import OpenAIClientSync
from conduit.domain.request.request import Request
from conduit.storage.odometer.usage import Usage
from conduit.domain.message.message import UserMessage
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
from pydantic import BaseModel
import pytest


class Frog(BaseModel):
    species: str
    age: int
    name: str
    occupation: str
    color: str
    legs: str
    continent: str


@pytest.fixture
def request_object():
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    return Request(model=model, messages=messages)


@pytest.fixture
def request_stream_object():
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "gpt-3.5-turbo-0125"
    return Request(model=model, messages=messages, stream=True)


@pytest.fixture
def request_structured_object():
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "gpt-4o"
    return Request(model=model, messages=messages, response_model=Frog)


# Google fixtures
@pytest.fixture
def google_request_object():
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "gemini-2.0-flash"
    return Request(model=model, messages=messages)


@pytest.fixture
def google_request_stream_object():
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "gemini-2.0-flash"
    return Request(model=model, messages=messages, stream=True)


@pytest.fixture
def google_request_structured_object():
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "gemini-2.0-flash"
    return Request(model=model, messages=messages, response_model=Frog)


# Ollama fixtures
@pytest.fixture
def ollama_request_object():
    user_message = UserMessage(content="Hello, how are you?")
    messages = [user_message]
    model = "llama3.1:latest"
    return Request(model=model, messages=messages)


@pytest.fixture
def ollama_request_stream_object():
    user_message = UserMessage(content="Tell me a story about a brave knight.")
    messages = [user_message]
    model = "llama3.1:latest"
    return Request(model=model, messages=messages, stream=True)


@pytest.fixture
def ollama_request_structured_object():
    user_message = UserMessage(
        content="Create a frog to delight me.",
    )
    messages = [user_message]
    model = "llama3.1:latest"
    return Request(model=model, messages=messages, response_model=Frog)


def test_openai_sync(request_object):
    client = OpenAIClientSync()
    response, usage = client.query(request_object)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_async(request_object):
    from conduit.core.model.clients.openai.client import OpenAIClientAsync

    client = OpenAIClientAsync()
    response, usage = await client.query(request_object)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


def test_openai_streaming(request_stream_object):
    client = OpenAIClientSync()
    response, usage = client.query(request_stream_object)
    assert response is not None
    assert isinstance(response, SyncStream)
    assert isinstance(usage, Usage)
    assert (
        usage.input_tokens == 0
    )  # (streaming requests typically do not count input tokens until completion)
    assert (
        usage.output_tokens == 0
    )  # (streaming requests typically do not count output tokens until completion)


@pytest.mark.asyncio
async def test_openai_streaming_async(request_stream_object):
    from conduit.core.model.clients.openai.client import OpenAIClientAsync

    client = OpenAIClientAsync()
    response, usage = await client.query(request_stream_object)
    assert response is not None
    assert isinstance(response, AsyncStream)
    assert isinstance(usage, Usage)
    assert (
        usage.input_tokens == 0
    )  # (streaming requests typically do not count input tokens until completion)
    assert (
        usage.output_tokens == 0
    )  # (streaming requests typically do not count output tokens until completion)


def test_openai_structured_response(request_structured_object):
    client = OpenAIClientSync()
    response, usage = client.query(request_structured_object)
    assert response is not None
    assert isinstance(response, Frog)
    assert hasattr(response, "species")
    assert hasattr(response, "age")
    assert hasattr(response, "name")
    assert hasattr(response, "occupation")
    assert hasattr(response, "color")
    assert hasattr(response, "legs")
    assert hasattr(response, "continent")
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_structured_response_async(request_structured_object):
    from conduit.core.model.clients.openai.client import OpenAIClientAsync

    client = OpenAIClientAsync()
    response, usage = await client.query(request_structured_object)
    assert response is not None
    assert isinstance(response, Frog)
    assert hasattr(response, "species")
    assert hasattr(response, "age")
    assert hasattr(response, "name")
    assert hasattr(response, "occupation")
    assert hasattr(response, "color")
    assert hasattr(response, "legs")
    assert hasattr(response, "continent")
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


# Google tests
def test_google_sync(google_request_object):
    from conduit.core.model.clients.google.client import GoogleClientSync

    client = GoogleClientSync()
    response, usage = client.query(google_request_object)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_google_async(google_request_object):
    from conduit.core.model.clients.google.client import GoogleClientAsync

    client = GoogleClientAsync()
    response, usage = await client.query(google_request_object)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


def test_google_streaming(google_request_stream_object):
    from conduit.core.model.clients.google.client import GoogleClientSync

    client = GoogleClientSync()
    response, usage = client.query(google_request_stream_object)
    assert response is not None
    assert isinstance(response, SyncStream)
    assert isinstance(usage, Usage)
    assert (
        usage.input_tokens == 0
    )  # (streaming requests typically do not count input tokens until completion)
    assert (
        usage.output_tokens == 0
    )  # (streaming requests typically do not count output tokens until completion)


@pytest.mark.asyncio
async def test_google_streaming_async(google_request_stream_object):
    from conduit.core.model.clients.google.client import GoogleClientAsync

    client = GoogleClientAsync()
    response, usage = await client.query(google_request_stream_object)
    assert response is not None
    assert isinstance(response, AsyncStream)
    assert isinstance(usage, Usage)
    assert (
        usage.input_tokens == 0
    )  # (streaming requests typically do not count input tokens until completion)
    assert (
        usage.output_tokens == 0
    )  # (streaming requests typically do not count output tokens until completion)


def test_google_structured_response(google_request_structured_object):
    from conduit.core.model.clients.google.client import GoogleClientSync

    client = GoogleClientSync()
    response, usage = client.query(google_request_structured_object)
    assert response is not None
    assert isinstance(response, Frog)
    assert hasattr(response, "species")
    assert hasattr(response, "age")
    assert hasattr(response, "name")
    assert hasattr(response, "occupation")
    assert hasattr(response, "color")
    assert hasattr(response, "legs")
    assert hasattr(response, "continent")
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_google_structured_response_async(google_request_structured_object):
    from conduit.core.model.clients.google.client import GoogleClientAsync

    client = GoogleClientAsync()
    response, usage = await client.query(google_request_structured_object)
    assert response is not None
    assert isinstance(response, Frog)
    assert hasattr(response, "species")
    assert hasattr(response, "age")
    assert hasattr(response, "name")
    assert hasattr(response, "occupation")
    assert hasattr(response, "color")
    assert hasattr(response, "legs")
    assert hasattr(response, "continent")
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


# Ollama tests
def test_ollama_sync(ollama_request_object):
    from conduit.core.model.clients.ollama.client import OllamaClientSync

    client = OllamaClientSync()
    response, usage = client.query(ollama_request_object)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_ollama_async(ollama_request_object):
    from conduit.core.model.clients.ollama.client import OllamaClientAsync

    client = OllamaClientAsync()
    response, usage = await client.query(ollama_request_object)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


def test_ollama_streaming(ollama_request_stream_object):
    from conduit.core.model.clients.ollama.client import OllamaClientSync

    client = OllamaClientSync()
    response, usage = client.query(ollama_request_stream_object)
    assert response is not None
    assert isinstance(response, SyncStream)
    assert isinstance(usage, Usage)
    assert (
        usage.input_tokens == 0
    )  # (streaming requests typically do not count input tokens until completion)
    assert (
        usage.output_tokens == 0
    )  # (streaming requests typically do not count output tokens until completion)


@pytest.mark.asyncio
async def test_ollama_streaming_async(ollama_request_stream_object):
    from conduit.core.model.clients.ollama.client import OllamaClientAsync

    client = OllamaClientAsync()
    response, usage = await client.query(ollama_request_stream_object)
    assert response is not None
    assert isinstance(response, AsyncStream)
    assert isinstance(usage, Usage)
    assert (
        usage.input_tokens == 0
    )  # (streaming requests typically do not count input tokens until completion)
    assert (
        usage.output_tokens == 0
    )  # (streaming requests typically do not count output tokens until completion)


def test_ollama_structured_response(ollama_request_structured_object):
    from conduit.core.model.clients.ollama.client import OllamaClientSync

    client = OllamaClientSync()
    response, usage = client.query(ollama_request_structured_object)
    assert response is not None
    assert isinstance(response, Frog)
    assert hasattr(response, "species")
    assert hasattr(response, "age")
    assert hasattr(response, "name")
    assert hasattr(response, "occupation")
    assert hasattr(response, "color")
    assert hasattr(response, "legs")
    assert hasattr(response, "continent")
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_ollama_structured_response_async(ollama_request_structured_object):
    from conduit.core.model.clients.ollama.client import OllamaClientAsync

    client = OllamaClientAsync()
    response, usage = await client.query(ollama_request_structured_object)
    assert response is not None
    assert isinstance(response, Frog)
    assert hasattr(response, "species")
    assert hasattr(response, "age")
    assert hasattr(response, "name")
    assert hasattr(response, "occupation")
    assert hasattr(response, "color")
    assert hasattr(response, "legs")
    assert hasattr(response, "continent")
    assert isinstance(usage, Usage)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
