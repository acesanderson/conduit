from __future__ import annotations

from unittest.mock import MagicMock, patch

from conduit.apps.cli.query.query_function import CLIQueryFunctionInputs, default_query_function


def make_inputs(**overrides) -> CLIQueryFunctionInputs:
    defaults = dict(
        project_name="test",
        query_input="hello",
        printer=MagicMock(),
        client_params={},
    )
    defaults.update(overrides)
    return CLIQueryFunctionInputs(**defaults)


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
