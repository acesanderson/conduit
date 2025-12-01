from conduit.core.model.clients.openai.client import OpenAIClientSync
from conduit.domain.request.request import Request
from conduit.storage.odometer.usage import Usage
from conduit.domain.message.message import UserMessage
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream
import pytest


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
