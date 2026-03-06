from __future__ import annotations
from unittest.mock import MagicMock, patch
import json
import logging
from conduit.apps.cli.handlers.base_handlers import BaseHandlers


def make_response(provider: str | None, citations: list[dict]) -> MagicMock:
    """Build a mock response object matching GenerationResponse shape."""
    metadata = {}
    if provider is not None:
        metadata["provider"] = provider
    metadata["citations"] = citations

    msg = MagicMock()
    msg.metadata = metadata

    response = MagicMock()
    response.message = msg
    return response


def make_printer() -> MagicMock:
    return MagicMock()


def test_handle_citations_noop_when_flag_false():
    """handle_citations does nothing when citations=False."""
    printer = make_printer()
    response = make_response("perplexity", [{"title": "T", "url": "U"}])
    BaseHandlers.handle_citations(response, citations=False, raw=False, printer=printer)
    printer.print_err.assert_not_called()
    printer.print_citations.assert_not_called()


def test_handle_citations_warns_when_non_perplexity_provider(caplog):
    """Prints red stderr warning when provider != 'perplexity'."""
    printer = make_printer()
    response = make_response("openai", [])

    with caplog.at_level(logging.WARNING, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[red]" in call_arg
    printer.print_citations.assert_not_called()
    assert any("openai" in r.message.lower() or "perplexity" in r.message.lower() for r in caplog.records)


def test_handle_citations_warns_when_provider_missing(caplog):
    """Prints red stderr warning when provider key is absent."""
    printer = make_printer()
    response = make_response(None, [])  # no provider key set

    with caplog.at_level(logging.WARNING, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[red]" in call_arg


def test_handle_citations_warns_yellow_when_citations_empty(caplog):
    """Prints yellow stderr warning when perplexity returns empty citations."""
    printer = make_printer()
    response = make_response("perplexity", [])

    with caplog.at_level(logging.DEBUG, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[yellow]" in call_arg
    printer.print_citations.assert_not_called()
    assert any("citation" in r.message.lower() for r in caplog.records)


def test_handle_citations_prints_formatted_when_raw_false(caplog):
    """Calls printer.print_citations in non-raw mode."""
    printer = make_printer()
    citations = [{"title": "X", "url": "https://x.com", "source": "", "date": ""}]
    response = make_response("perplexity", citations)

    with caplog.at_level(logging.DEBUG, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_citations.assert_called_once_with(citations)
    printer.print_err.assert_not_called()
    assert any("1" in r.message for r in caplog.records)


def test_handle_citations_prints_json_when_raw_true(caplog):
    """click.echo(json.dumps(citations)) called in raw mode, not print_citations."""
    printer = make_printer()
    citations = [{"title": "X", "url": "https://x.com", "source": "", "date": ""}]
    response = make_response("perplexity", citations)

    with caplog.at_level(logging.DEBUG, logger="conduit.apps.cli.handlers.base_handlers"):
        with patch("conduit.apps.cli.handlers.base_handlers.click.echo") as mock_echo:
            BaseHandlers.handle_citations(response, citations=True, raw=True, printer=printer)

    mock_echo.assert_called_once()
    echoed = mock_echo.call_args[0][0]
    parsed = json.loads(echoed)
    assert parsed == citations
    printer.print_citations.assert_not_called()


def test_handle_citations_handles_none_metadata():
    """handle_citations does not crash when message.metadata is None."""
    msg = MagicMock()
    msg.metadata = None
    response = MagicMock()
    response.message = msg
    printer = make_printer()
    # Should not raise; should warn as non-perplexity (metadata is empty after or-guard)
    BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)
    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[red]" in call_arg


def test_handle_citations_works_with_conversation_shape():
    """handle_citations works when response has .last (Conversation shape, not .message)."""
    # Simulate a Conversation object: no .message attribute, has .last
    msg = MagicMock(spec=[])  # spec=[] means no attributes by default
    msg.metadata = {
        "provider": "perplexity",
        "citations": [{"title": "T", "url": "https://t.com", "source": "", "date": ""}],
    }

    response = MagicMock(spec=[])  # no .message attribute
    response.last = msg

    printer = make_printer()
    BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_citations.assert_called_once()
    printer.print_err.assert_not_called()
