from __future__ import annotations
import sys
from unittest.mock import patch, MagicMock
from conduit.apps.cli.utils.printer import Printer


def make_printer(raw: bool = False, is_tty: bool = True) -> Printer:
    with patch("conduit.apps.cli.utils.printer.IS_TTY", is_tty):
        return Printer(raw=raw)


def test_print_err_goes_to_stderr_in_tty():
    """print_err writes to stderr in TTY mode."""
    printer = make_printer(is_tty=True)
    with patch.object(printer, "_err_console") as mock_console:
        printer.print_err("[red]warning[/red]")
        mock_console.print.assert_called_once_with("[red]warning[/red]")


def test_print_err_goes_to_stderr_in_pipe():
    """print_err writes to stderr even when piped (not TTY)."""
    printer = make_printer(is_tty=False)
    with patch.object(printer, "_err_console") as mock_console:
        printer.print_err("[red]warning[/red]")
        mock_console.print.assert_called_once_with("[red]warning[/red]")


def test_print_citations_renders_numbered_list_in_tty():
    """print_citations renders a numbered list via print_pretty in TTY mode."""
    printer = make_printer(is_tty=True, raw=False)
    citations = [
        {"title": "Article One", "url": "https://one.com", "source": "", "date": ""},
        {"title": "Article Two", "url": "https://two.com", "source": "", "date": ""},
    ]
    with patch.object(printer, "print_pretty") as mock_pp:
        printer.print_citations(citations)
    calls = [str(c) for c in mock_pp.call_args_list]
    assert any("1." in c and "Article One" in c and "https://one.com" in c for c in calls)
    assert any("2." in c and "Article Two" in c and "https://two.com" in c for c in calls)


def test_print_citations_handles_missing_keys_gracefully():
    """print_citations doesn't crash on citations with missing title or url."""
    printer = make_printer(is_tty=True, raw=False)
    citations = [
        {"title": "", "url": "", "source": "", "date": ""},   # both empty — should be skipped
        {"title": "Only Title", "url": "", "source": "", "date": ""},
        {"url": "https://only-url.com"},                       # no title key at all
    ]
    with patch.object(printer, "print_pretty") as mock_pp:
        printer.print_citations(citations)
    # header ("") + "[bold]Sources[/bold]" + 2 non-empty entries = at least 4 calls
    assert mock_pp.call_count >= 2
    calls = [str(c) for c in mock_pp.call_args_list]
    # the all-empty entry must not appear in any call
    assert not any("" == str(c) and "title" in c for c in calls)
    # the two non-empty entries must appear
    assert any("Only Title" in c for c in calls)
    assert any("only-url.com" in c for c in calls)


def test_print_citations_silent_in_pipe_mode():
    """print_citations does nothing in pipe mode (data mode handles JSON separately)."""
    printer = make_printer(is_tty=False, raw=False)
    with patch.object(printer, "print_pretty") as mock_pp:
        printer.print_citations([{"title": "X", "url": "y"}])
    mock_pp.assert_not_called()
