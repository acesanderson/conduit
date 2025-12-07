from typing import Any, override
from rich.console import Console
from rich.status import Status

from conduit.utils.progress.protocol import DisplayHandler
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.progress.rich_formatters import (
    format_response_rich,
    format_error_rich,
)


class RichProgressHandler(DisplayHandler):
    """
    Stateful progress handler for Rich TUI.
    Manages the active spinner and renders formatted panels.
    """

    def __init__(self, console: Console):
        self.console = console
        # We hold the active status object so we can stop it later
        self._status: Status | None = None

    def _stop_spinner(self):
        """Helper to cleanly stop any active spinner."""
        if self._status:
            self._status.stop()
            self._status = None

    @override
    def show_spinner(
        self, model_name: str, query_preview: str, verbosity: Verbosity
    ) -> None:
        if verbosity == Verbosity.SILENT:
            return

        # Stop any existing spinner to avoid visual glitches
        self._stop_spinner()

        if verbosity >= Verbosity.PROGRESS:
            # Create and start the spinner
            # We don't use 'with' here because the context manager logic
            # lives in the middleware, not the handler.
            status_text = f"[bold gold1]{model_name}[/bold gold1] | {query_preview}"
            self._status = self.console.status(status_text, spinner="dots")
            self._status.start()

    @override
    def show_complete(
        self,
        model_name: str,
        query_preview: str,
        duration: float,
        verbosity: Verbosity,
        response_obj: Any | None = None,
    ) -> None:
        self._stop_spinner()

        if verbosity == Verbosity.SILENT:
            return

        if verbosity >= Verbosity.PROGRESS:
            # 1. The One-Liner
            self.console.print(
                f"[green]✓[/green] [bold white]{model_name}[/bold white] | {query_preview} | [dim]({duration:.2f}s)[/dim]"
            )

            # 2. The Detail Panel (delegated to formatter)
            if response_obj and verbosity >= Verbosity.SUMMARY:
                panel = format_response_rich(response_obj, verbosity)
                if panel:
                    self.console.print(panel)

    @override
    def show_cached(
        self, model_name: str, query_preview: str, duration: float, verbosity: Verbosity
    ) -> None:
        self._stop_spinner()

        if verbosity == Verbosity.SILENT:
            return

        if verbosity >= Verbosity.PROGRESS:
            self.console.print(
                f"⚡ [bold cyan]{model_name}[/bold cyan] | {query_preview} | [cyan]Cached[/cyan] [dim]({duration:.3f}s)[/dim]"
            )

    @override
    def show_failed(
        self,
        model_name: str,
        query_preview: str,
        error: str,
        verbosity: Verbosity,
        error_obj: Any | None = None,
    ) -> None:
        self._stop_spinner()

        if verbosity == Verbosity.SILENT:
            return

        if verbosity >= Verbosity.PROGRESS:
            self.console.print(
                f"[red]✗[/red] [bold white]{model_name}[/bold white] | {query_preview} | [red]Failed: {error}[/red]"
            )

            if error_obj and verbosity >= Verbosity.SUMMARY:
                panel = format_error_rich(error_obj, verbosity)
                if panel:
                    self.console.print(panel)
