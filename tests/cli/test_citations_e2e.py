from __future__ import annotations
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from conduit.apps.cli.cli_class import ConduitCLI
from conduit.apps.cli.commands.base_commands import BaseCommands


def make_cli_with_mock_qf(provider: str = "perplexity", citations: list | None = None):
    """Build a test CLI with a mock query function returning a mock response."""
    if citations is None:
        citations = [{"title": "Src", "url": "https://src.com", "source": "", "date": ""}]

    mock_qf = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "The answer."
    # Use GenerationResponse shape: .message.metadata
    mock_response.message.metadata = {
        "provider": provider,
        "citations": citations,
    }
    mock_qf.return_value = mock_response

    cli_app = ConduitCLI(query_function=mock_qf)
    cli_app.attach(BaseCommands())
    return cli_app.cli, mock_qf


def test_citations_flag_appears_in_help():
    """--citations and -C appear in `conduit query --help`."""
    cli, _ = make_cli_with_mock_qf()
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])
    assert "--citations" in result.output or "-C" in result.output


def test_citations_flag_sets_client_params():
    """--citations flag causes client_params={"return_citations": True} in inputs."""
    cli, mock_qf = make_cli_with_mock_qf()
    runner = CliRunner()
    runner.invoke(cli, ["query", "--citations", "what is rust"])
    assert mock_qf.called
    inputs = mock_qf.call_args[0][0]
    assert inputs.client_params == {"return_citations": True}


def test_no_citations_flag_leaves_client_params_empty():
    """Without --citations, client_params is empty."""
    cli, mock_qf = make_cli_with_mock_qf()
    runner = CliRunner()
    runner.invoke(cli, ["query", "what is rust"])
    inputs = mock_qf.call_args[0][0]
    assert inputs.client_params == {}


def test_citations_short_form():
    """-C short form works identically to --citations."""
    cli, mock_qf = make_cli_with_mock_qf()
    runner = CliRunner()
    runner.invoke(cli, ["query", "-C", "what is rust"])
    inputs = mock_qf.call_args[0][0]
    assert inputs.client_params == {"return_citations": True}


def test_citations_non_perplexity_exits_cleanly():
    """Non-Perplexity response: exit code 0, red warning emitted, no exception raised.

    CliRunner in Click 8.x does not support mix_stderr=False, so stderr capture via
    result.stderr is unavailable. Rich Console also writes directly to sys.stderr,
    bypassing CliRunner's stdout capture entirely. Instead, we patch printer.print_err
    to assert the warning was actually emitted.
    """
    cli, _ = make_cli_with_mock_qf(provider="openai", citations=[])
    runner = CliRunner()
    from conduit.apps.cli.utils.printer import Printer
    with patch.object(Printer, "print_err") as mock_print_err:
        result = runner.invoke(cli, ["query", "--citations", "hello"])
    assert result.exit_code == 0
    assert result.exception is None
    # Red warning must have been emitted referencing perplexity or citations
    mock_print_err.assert_called_once()
    warning_text: str = mock_print_err.call_args[0][0].lower()
    assert "perplexity" in warning_text or "citations" in warning_text


def test_no_citations_flag_means_handle_citations_called_with_false():
    """Without --citations, handle_citations is called with citations=False (no-op)."""
    cli, _ = make_cli_with_mock_qf()
    runner = CliRunner()
    with patch("conduit.apps.cli.handlers.base_handlers.BaseHandlers.handle_citations") as mock_hc:
        runner.invoke(cli, ["query", "what is rust"])
    mock_hc.assert_called_once()
    call_args = mock_hc.call_args
    assert call_args.kwargs.get("citations") is False


def test_citations_flag_calls_handle_citations():
    """With --citations, handle_citations is called with citations=True."""
    cli, _ = make_cli_with_mock_qf()
    runner = CliRunner()
    with patch("conduit.apps.cli.handlers.base_handlers.BaseHandlers.handle_citations") as mock_hc:
        runner.invoke(cli, ["query", "--citations", "what is rust"])
    mock_hc.assert_called_once()
    call_args = mock_hc.call_args
    # handle_citations(response, citations=True, raw=False, printer=...)
    assert call_args.kwargs.get("citations") is True
