import sys
from typing import Any, override
from datetime import datetime
from conduit.utils.progress.protocol import DisplayHandler
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.progress.plain_formatters import (
    format_response_plain,
    format_error_plain,
)


class PlainProgressHandler(DisplayHandler):
    """
    Append-only progress handler.
    Writes to STDERR so it doesn't corrupt piped output.
    """

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("[%H:%M:%S]")

    @override
    def show_spinner(
        self, model_name: str, query_preview: str, verbosity: Verbosity
    ) -> None:
        if verbosity >= Verbosity.PROGRESS:
            ts = self._get_timestamp()
            # Added file=sys.stderr and flush=True
            print(
                f"{ts} [{model_name}] Starting: {query_preview}",
                file=sys.stderr,
                flush=True,
            )

    @override
    def show_complete(
        self,
        model_name: str,
        query_preview: str,
        duration: float,
        verbosity: Verbosity,
        response_obj: Any | None = None,
    ) -> None:
        if verbosity >= Verbosity.PROGRESS:
            ts = self._get_timestamp()
            print(
                f"{ts} [{model_name}] Completed ({duration:.2f}s)",
                file=sys.stderr,
                flush=True,
            )

        if response_obj and verbosity >= Verbosity.SUMMARY:
            text = format_response_plain(response_obj, verbosity)
            if text:
                print(text, file=sys.stderr, flush=True)

    @override
    def show_cached(
        self, model_name: str, query_preview: str, duration: float, verbosity: Verbosity
    ) -> None:
        if verbosity >= Verbosity.PROGRESS:
            ts = self._get_timestamp()
            print(
                f"{ts} [{model_name}] Cache Hit ({duration:.3f}s): {query_preview}",
                file=sys.stderr,
                flush=True,
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
        if verbosity >= Verbosity.PROGRESS:
            ts = self._get_timestamp()
            print(f"{ts} [{model_name}] Failed: {error}", file=sys.stderr, flush=True)

        if error_obj and verbosity >= Verbosity.SUMMARY:
            text = format_error_plain(error_obj, verbosity)
            if text:
                print(text, file=sys.stderr, flush=True)
