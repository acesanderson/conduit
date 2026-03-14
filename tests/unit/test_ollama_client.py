from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.core.clients.ollama.client import OllamaClient
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import GenerationRequest


@pytest.fixture
def client():
    """Minimal OllamaClient with all I/O mocked out."""
    with patch.object(OllamaClient, "__init__", lambda self: None):
        c = OllamaClient()
        c._ollama_context_sizes = MagicMock(__getitem__=MagicMock(return_value=32768))
        c._client = MagicMock()           # instructor-wrapped client
        c._raw_client = AsyncMock()       # raw AsyncOpenAI client
        return c


def make_params(**kwargs) -> GenerationParams:
    """Construct GenerationParams bypassing model validation."""
    defaults = dict(
        model="gpt-oss:latest",
        output_type="structured_response",
        response_model=None,
        response_model_schema=None,
        temperature=0.0,
        client_params={"num_ctx": 16384},
    )
    defaults.update(kwargs)
    return GenerationParams.model_construct(**defaults)


# Fulfills AC 5: _generate_structured_response with both None raises ValueError
# before any network call.
@pytest.mark.asyncio
async def test_structured_response_raises_when_no_model_and_no_schema(client):
    request = MagicMock()
    request.params = make_params(
        output_type="structured_response",
        response_model=None,
        response_model_schema=None,
    )

    with pytest.raises(ValueError, match="structured_response requires"):
        await client._generate_structured_response(request)

    client._raw_client.chat.completions.create.assert_not_called()
    client._client.chat.completions.create_with_completion.assert_not_called()


# Fulfills AC 3: _convert_request with schema path conditions produces
# payload_dict["response_format"]["type"] == "json_schema" and schema matches.
def test_convert_request_injects_response_format_for_schema_path(client):
    from pydantic import BaseModel
    from enum import Enum

    class Color(str, Enum):
        red = "red"
        blue = "blue"

    class Widget(BaseModel):
        name: str
        color: Color

    schema = Widget.model_json_schema()
    params = make_params(
        output_type="structured_response",
        response_model=None,
        response_model_schema=schema,
    )
    request = MagicMock()
    request.params = params
    request.messages = []

    with patch.object(client, "_convert_messages", return_value=[]):
        payload = client._convert_request(request)

    payload_dict = payload.model_dump(exclude_none=True)

    assert "response_format" in payload_dict
    assert payload_dict["response_format"]["type"] == "json_schema"
    assert payload_dict["response_format"]["json_schema"]["schema"] == schema
    assert payload_dict["response_format"]["json_schema"]["name"] == "Widget"
    assert payload_dict["response_format"]["json_schema"]["strict"] is True


# Fulfills AC 4: _convert_request with output_type="text" and response_model_schema
# set does NOT produce a response_format key — guards against poisoning text calls.
def test_convert_request_no_response_format_for_text_output(client):
    from pydantic import BaseModel

    class Widget(BaseModel):
        name: str

    schema = Widget.model_json_schema()
    params = make_params(
        output_type="text",
        response_model=None,
        response_model_schema=schema,   # schema present but output_type is text
    )
    request = MagicMock()
    request.params = params
    request.messages = []

    with patch.object(client, "_convert_messages", return_value=[]):
        payload = client._convert_request(request)

    payload_dict = payload.model_dump(exclude_none=True)
    assert "response_format" not in payload_dict


# Fulfills AC 7: schema path sets message.parsed = None in the GenerationResponse.
@pytest.mark.asyncio
async def test_schema_path_returns_parsed_none(client):
    from pydantic import BaseModel

    class Widget(BaseModel):
        name: str

    schema = Widget.model_json_schema()
    params = make_params(
        output_type="structured_response",
        response_model=None,
        response_model_schema=schema,
    )
    request = GenerationRequest.model_construct(params=params, messages=[], options=None)

    # Mock the raw_client response
    mock_choice = MagicMock()
    mock_choice.message.content = '{"name": "thing"}'
    mock_choice.finish_reason = "stop"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.model = "gpt-oss:latest"
    mock_completion.usage.prompt_tokens = 10
    mock_completion.usage.completion_tokens = 5
    client._raw_client.chat.completions.create = AsyncMock(
        return_value=mock_completion
    )

    with patch.object(client, "_convert_request") as mock_convert:
        mock_payload = MagicMock()
        mock_payload.model_dump.return_value = {"model": "gpt-oss:latest", "messages": []}
        mock_convert.return_value = mock_payload
        response = await client._generate_structured_response(request)

    assert response.message.parsed is None
    assert response.message.content == '{"name": "thing"}'


# Fulfills AC 6: when response_model is non-None (instructor path), self._client
# is called, not self._raw_client. Guards against instructor path regression.
@pytest.mark.asyncio
async def test_instructor_path_uses_client_not_raw_client(client):
    from pydantic import BaseModel

    class Widget(BaseModel):
        name: str

    params = make_params(
        output_type="structured_response",
        response_model=Widget,
        response_model_schema=Widget.model_json_schema(),
    )
    request = GenerationRequest.model_construct(params=params, messages=[], options=None)

    # Mock instructor response
    mock_widget = Widget(name="sprocket")
    mock_choice = MagicMock()
    mock_choice.message.content = '{"name": "sprocket"}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.model = "gpt-oss:latest"
    mock_completion.usage.prompt_tokens = 10
    mock_completion.usage.completion_tokens = 5

    client._client.chat = MagicMock()
    client._client.chat.completions = MagicMock()
    client._client.chat.completions.create_with_completion = AsyncMock(
        return_value=(mock_widget, mock_completion)
    )

    with patch.object(client, "_convert_request") as mock_convert:
        mock_payload = MagicMock()
        mock_payload.model_dump.return_value = {"model": "gpt-oss:latest", "messages": []}
        mock_convert.return_value = mock_payload
        await client._generate_structured_response(request)

    client._client.chat.completions.create_with_completion.assert_called_once()
    client._raw_client.chat.completions.create.assert_not_called()
