from __future__ import annotations

from unittest.mock import MagicMock, patch

from conduit.apps.cli.query.query_function import CLIQueryFunctionInputs, default_query_function
from conduit.domain.message.message import TextContent, ImageContent


def make_inputs(**overrides) -> CLIQueryFunctionInputs:
    defaults = dict(
        project_name="test",
        query_input="hello",
        printer=MagicMock(),
        client_params={},
    )
    defaults.update(overrides)
    return CLIQueryFunctionInputs(**defaults)


def test_inputs_image_path_defaults_to_none():
    """image_path field exists and defaults to None."""
    inputs = make_inputs()
    assert inputs.image_path is None


def test_inputs_accepts_image_path():
    """image_path accepts a string path."""
    inputs = make_inputs(image_path="/tmp/test.png")
    assert inputs.image_path == "/tmp/test.png"


def test_image_query_builds_multimodal_usermessage(tmp_path):
    """AC1: image branch builds UserMessage with [TextContent, ImageContent] in that order."""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    inputs = make_inputs(query_input="describe this", image_path=str(img))

    captured = {}

    def fake_pipe_sync(conversation):
        captured["conversation"] = conversation
        return MagicMock()

    mock_conduit = MagicMock()
    mock_conduit.pipe_sync.side_effect = fake_pipe_sync

    with patch(
        "conduit.apps.cli.query.query_function.ConduitSync",
        return_value=mock_conduit,
    ):
        default_query_function(inputs)

    conversation = captured["conversation"]
    user_msgs = [m for m in conversation.messages if hasattr(m, "role") and str(m.role) == "Role.USER"]
    assert len(user_msgs) == 1
    content = user_msgs[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], TextContent), "TextContent must be first"
    assert isinstance(content[1], ImageContent), "ImageContent must be second"
    assert content[0].text == "describe this"


def test_inputs_has_client_params_field():
    """CLIQueryFunctionInputs accepts client_params kwarg."""
    inputs = make_inputs(client_params={"return_citations": True})
    assert inputs.client_params == {"return_citations": True}


def test_inputs_client_params_defaults_to_empty_dict():
    """client_params defaults to empty dict, not None."""
    inputs = make_inputs()
    assert inputs.client_params == {}


def test_default_query_function_passes_client_params_to_conduit():
    """default_query_function passes client_params to ConduitSync.create()."""
    inputs = make_inputs(client_params={"return_citations": True})

    mock_response = MagicMock()
    mock_conduit = MagicMock(return_value=mock_response)

    with patch(
        "conduit.core.conduit.conduit_sync.ConduitSync.create",
        return_value=mock_conduit,
    ) as mock_create:
        default_query_function(inputs)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs.get("client_params") == {"return_citations": True}


def test_default_query_function_omits_client_params_when_empty():
    """When client_params is empty, None is passed to avoid polluting GenerationParams."""
    inputs = make_inputs(client_params={})

    mock_response = MagicMock()
    mock_conduit = MagicMock(return_value=mock_response)

    with patch(
        "conduit.core.conduit.conduit_sync.ConduitSync.create",
        return_value=mock_conduit,
    ) as mock_create:
        default_query_function(inputs)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs.get("client_params") is None
