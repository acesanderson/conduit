from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.core.clients.ollama.client import OllamaClient
from conduit.domain.request.generation_params import GenerationParams


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
