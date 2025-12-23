"""
Module for displaying data and UI elements conditionally based on TTY status.
- If stdout is a TTY: show Rich UI to stderr, suppress data unless --raw
- If stdout is piped/redirected: emit data to stdout, suppress UI
This mirrors best practice for POSIX-friendly CLIs.
"""

import sys
from contextlib import nullcontext
from signal import signal, SIGPIPE, SIG_DFL
from rich.console import Console

# Treat broken pipes cleanly (avoid stack traces in pipelines)
_ = signal(SIGPIPE, SIG_DFL)

IS_TTY = sys.stdout.isatty()


class Printer:
    def __init__(self, raw: bool = False):
        """
        Initialize IO policy based on TTY status and raw flag.
        """
        self.emit_data = (not IS_TTY) or raw  # pipe/redirect OR --raw
        self.emit_ui = IS_TTY and (not raw)  # bare terminal AND not --raw
        self.ui = Console(file=sys.stderr) if self.emit_ui else None
        self._write = sys.stdout.write

    def set_raw(self, raw: bool):
        """
        Update IO policy based on raw flag.
        """
        self.emit_data = (not IS_TTY) or raw
        self.emit_ui = IS_TTY and (not raw)
        self.ui = Console(file=sys.stderr) if self.emit_ui else None

    def print_raw(self, s: str = ""):
        """
        Data stream for piping/redirecting (stdout).
        """
        if self.emit_data:
            self._write(s)
            if s and not s.endswith("\n"):
                self._write("\n")

    def print_pretty(self, *args, **kwargs) -> None:
        """
        Human-facing UI (stderr via Rich).
        """
        if self.ui:
            self.ui.print(*args, **kwargs)

    def status(self, *args, **kwargs):
        """
        Context manager for spinners/status messages.
        Disabled when UI is off.
        """
        if self.ui:
            return self.ui.status(*args, **kwargs)
        return nullcontext()

    def print_markdown(self, markdown_string: str, add_rule: bool = True):
        """
        Unified Markdown printer:
        - If piping/redirecting (emit_data): write plain Markdown to stdout.
        - If TTY UI (emit_ui): render via Rich Markdown on stderr.
        """
        if self.emit_data:
            self.print_raw(markdown_string)
            return
        if self.ui:
            from rich.markdown import Markdown

            md = markdown_string
            if add_rule:
                border = "-" * 100
                md = f"{border}\n{markdown_string}\n\n{border}"
            self.ui.print(Markdown(md))
