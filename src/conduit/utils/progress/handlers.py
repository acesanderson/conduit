"""
Enhanced progress handlers with verbosity support.
Maintains backwards compatibility while adding verbosity-aware methods.
"""

from conduit.utils.progress.verbosity import Verbosity
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class RichProgressHandler:
    """Rich-based progress handler with spinners and colors - now verbosity-aware"""

    def __init__(self, console):
        self.console = console
        self.concurrent_mode = False
        self.concurrent_line_printed = False

    # Enhanced individual operation methods with verbosity support
    def show_spinner(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        """Show Rich spinner with live status - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            self.console.print(
                f"⠋ {model_name} | {query_preview}",
                end="\r",
                highlight=False,
                soft_wrap=True,
            )
        # Higher verbosity levels will be handled in future phases

    def show_complete(
        self,
        model_name,
        query_preview,
        duration,
        verbosity=Verbosity.PROGRESS,
        response_obj=None,
    ):
        """Update same line with green checkmark - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            self.console.print(
                f"✓ {model_name} | {query_preview} | ({duration:.1f}s)", style="green"
            )
        elif verbosity >= Verbosity.SUMMARY and response_obj != None:
            # New behavior - use object display methods
            # First show the progress line
            self.console.print(
                f"✓ {model_name} | {query_preview} | ({duration:.1f}s)", style="green"
            )
            # Then show the detailed response
            rich_content = response_obj.to_rich(verbosity)
            if rich_content:
                self.console.print()  # Add spacing
                self.console.print(rich_content)

    def show_canceled(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        """Update same line with warning - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            self.console.print(
                f"⚠ {model_name} | {query_preview} | Canceled", style="yellow"
            )
        # Higher verbosity levels will be handled in future phases

    def show_cached(
        self, model_name, query_preview, duration, verbosity=Verbosity.PROGRESS
    ):
        """Show cache hit with lightning symbol - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            self.console.print(
                f"⚡ {model_name} | {query_preview} | Cached ({duration:.1f}s)",
                style="cyan",
            )
        # Higher verbosity levels will be handled in future phases

    def show_failed(
        self,
        model_name,
        query_preview,
        error,
        verbosity=Verbosity.PROGRESS,
        error_obj=None,
    ):
        """Update same line with error - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            self.console.print(
                f"✗ {model_name} | {query_preview} | Failed: {error}", style="red"
            )
        elif verbosity >= Verbosity.SUMMARY and error_obj:
            # New behavior - use object display methods
            # First show the progress line
            self.console.print(
                f"✗ {model_name} | {query_preview} | Failed: {error}", style="red"
            )
            # Then show the detailed error
            rich_content = error_obj.to_rich(verbosity)
            if rich_content:
                self.console.print()  # Add spacing
                self.console.print(rich_content)

    # Concurrent operation methods (unchanged for now)
    def handle_concurrent_start(self, total: int):
        """Handle start of concurrent operations"""
        self.concurrent_mode = True
        self.concurrent_line_printed = False
        self.console.print(
            f"⠋ Running {total} concurrent requests...", end="\r", highlight=False
        )

    def update_concurrent_progress(
        self, completed: int, total: int, running: int, failed: int, elapsed: float
    ):
        """Update live concurrent progress"""
        if not self.concurrent_mode:
            return

        # Only show progress updates every 0.5 seconds to avoid spam
        current_time = time.time()
        if not hasattr(self, "_last_update") or current_time - self._last_update > 0.5:
            self._last_update = current_time

            progress_text = f"⠋ Progress: {completed}/{total} complete | {running} running | {failed} failed | {elapsed:.1f}s elapsed"
            self.console.print(progress_text, end="\r", highlight=False)

    def handle_concurrent_complete(self, successful: int, total: int, duration: float):
        """Handle completion of all concurrent operations"""
        self.concurrent_mode = False

        if successful == total:
            self.console.print(
                f"[green]✓[/green] All requests complete: {successful}/{total} successful in {duration:.1f}s"
            )
        else:
            failed = total - successful
            self.console.print(
                f"[yellow]✓[/yellow] All requests complete: {successful}/{total} successful, {failed} failed in {duration:.1f}s"
            )

    # Backwards compatibility methods (now verbosity-aware with defaults)
    def emit_started(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        self.show_spinner(model_name, query_preview, verbosity=verbosity)

    def emit_complete(
        self, model_name, query_preview, duration, verbosity=Verbosity.PROGRESS
    ):
        self.show_complete(model_name, query_preview, duration, verbosity=verbosity)

    def emit_canceled(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        self.show_canceled(model_name, query_preview, verbosity=verbosity)

    def emit_cached(
        self, model_name, query_preview, duration, verbosity=Verbosity.PROGRESS
    ):
        self.show_cached(model_name, query_preview, duration, verbosity=verbosity)

    def emit_failed(
        self, model_name, query_preview, error, verbosity=Verbosity.PROGRESS
    ):
        self.show_failed(model_name, query_preview, error, verbosity=verbosity)


class PlainProgressHandler:
    """Simple progress handler for environments without Rich - now verbosity-aware"""

    def __init__(self):
        self.concurrent_mode = False

    # Enhanced individual operation methods with verbosity support
    def show_spinner(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        """Show starting state (plain text - no spinner) - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Starting: {query_preview}")
        # Higher verbosity levels will be handled in future phases

    def show_complete(
        self,
        model_name,
        query_preview,
        duration,
        verbosity=Verbosity.PROGRESS,
        response_obj=None,
    ):
        """Show completion on new line - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Complete: ({duration:.1f}s)")
        elif verbosity >= Verbosity.SUMMARY and response_obj:
            # New behavior - use object display methods
            # First show the progress line
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Complete: ({duration:.1f}s)")
            # Then show the detailed response
            plain_content = response_obj.to_plain(verbosity)
            if plain_content:
                print(plain_content)

    def show_canceled(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        """Show cancellation on new line - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Canceled")
        # Higher verbosity levels will be handled in future phases

    def show_cached(
        self, model_name, query_preview, duration, verbosity=Verbosity.PROGRESS
    ):
        """Show cache hit in plain text - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Cache hit: {query_preview}")
        # Higher verbosity levels will be handled in future phases

    def show_failed(
        self,
        model_name,
        query_preview,
        error,
        verbosity=Verbosity.PROGRESS,
        error_obj=None,
    ):
        """Show failure on new line - verbosity aware"""
        if verbosity == Verbosity.SILENT:
            return

        if self.concurrent_mode:
            return

        if verbosity == Verbosity.PROGRESS:
            # Current behavior for backwards compatibility
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Failed: {error}")
        elif verbosity >= Verbosity.SUMMARY and error_obj:
            # New behavior - use object display methods
            # First show the progress line
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{model_name}] Failed: {error}")
            # Then show the detailed error
            plain_content = error_obj.to_plain(verbosity)
            if plain_content:
                print(plain_content)

    # Concurrent operation methods (unchanged for now)
    def handle_concurrent_start(self, total: int):
        """Handle start of concurrent operations"""
        self.concurrent_mode = True
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Starting: {total} concurrent requests")

    def update_concurrent_progress(
        self, completed: int, total: int, running: int, failed: int, elapsed: float
    ):
        """Plain console doesn't show live updates - too noisy"""
        pass  # Plain console shows only start/end messages

    def handle_concurrent_complete(self, successful: int, total: int, duration: float):
        """Handle completion of all concurrent operations"""
        self.concurrent_mode = False
        timestamp = datetime.now().strftime("%H:%M:%S")

        if successful == total:
            print(
                f"[{timestamp}] All requests complete: {successful}/{total} successful in {duration:.1f}s"
            )
        else:
            failed = total - successful
            print(
                f"[{timestamp}] All requests complete: {successful}/{total} successful, {failed} failed in {duration:.1f}s"
            )

    # Backwards compatibility methods (now verbosity-aware with defaults)
    def emit_started(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        self.show_spinner(model_name, query_preview, verbosity=verbosity)

    def emit_complete(
        self, model_name, query_preview, duration, verbosity=Verbosity.PROGRESS
    ):
        self.show_complete(model_name, query_preview, duration, verbosity=verbosity)

    def emit_canceled(self, model_name, query_preview, verbosity=Verbosity.PROGRESS):
        self.show_canceled(model_name, query_preview, verbosity=verbosity)

    def emit_cached(
        self, model_name, query_preview, duration, verbosity=Verbosity.PROGRESS
    ):
        self.show_cached(model_name, query_preview, duration, verbosity=verbosity)

    def emit_failed(
        self, model_name, query_preview, error, verbosity=Verbosity.PROGRESS
    ):
        self.show_failed(model_name, query_preview, error, verbosity=verbosity)
