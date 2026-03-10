from __future__ import annotations
import json
from unittest.mock import MagicMock, patch
import pytest
from conduit.apps.cli.handlers.batch_handlers import BatchHandlers


def _make_mock_conversation(text: str) -> MagicMock:
    conv = MagicMock()
    conv.content = text
    return conv


def _make_printer() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_batch_run():
    """Patch ConduitBatchSync.create so no real LLM calls happen."""
    conversations = [
        _make_mock_conversation("Response A"),
        _make_mock_conversation("Response B"),
    ]
    mock_instance = MagicMock()
    mock_instance.run.return_value = conversations
    with patch(
        "conduit.apps.cli.handlers.batch_handlers.ConduitBatchSync"
    ) as mock_cls:
        mock_cls.create.return_value = mock_instance
        yield mock_cls, mock_instance


PROMPTS = ["Prompt one", "Prompt two"]


def test_handle_batch_calls_batch_sync(mock_batch_run):
    mock_cls, mock_instance = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=False,
        printer=printer,
    )
    mock_cls.create.assert_called_once()
    call_kwargs = mock_cls.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    mock_instance.run.assert_called_once_with(
        prompt_strings_list=PROMPTS, max_concurrent=None
    )


def test_handle_batch_pretty_mode_prints_headers(mock_batch_run):
    _, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=False,
        printer=printer,
    )
    # Should call print_pretty for headers and print_markdown for bodies
    assert printer.print_pretty.call_count == 2
    assert printer.print_markdown.call_count == 2


def test_handle_batch_raw_mode(mock_batch_run, capsys):
    _, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=True,
        as_json=False,
        printer=printer,
    )
    captured = capsys.readouterr()
    assert "Response A" in captured.out
    assert "Response B" in captured.out
    assert "---" in captured.out
    # Pretty mode should NOT have been called
    printer.print_pretty.assert_not_called()


def test_handle_batch_json_mode(mock_batch_run, capsys):
    _, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=True,
        printer=printer,
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 2
    assert data[0]["index"] == 0
    assert data[0]["prompt"] == "Prompt one"
    assert data[0]["response"] == "Response A"
    assert data[1]["index"] == 1
    assert data[1]["response"] == "Response B"


def test_handle_batch_temperature_forwarded(mock_batch_run):
    mock_cls, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=["Q"],
        model="gpt-4o",
        temperature=0.3,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=False,
        printer=printer,
    )
    call_kwargs = mock_cls.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.3


def test_handle_batch_max_concurrent_forwarded(mock_batch_run):
    _, mock_instance = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=["Q"],
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=3,
        raw=False,
        as_json=False,
        printer=printer,
    )
    mock_instance.run.assert_called_once_with(
        prompt_strings_list=["Q"], max_concurrent=3
    )
