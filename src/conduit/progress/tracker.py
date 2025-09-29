from conduit.logs.logging_config import get_logger
from datetime import datetime
from pydantic import BaseModel
from typing import Protocol
from datetime import datetime
import time

logger = get_logger(__name__)


class ProgressEvent(BaseModel):
    pass


class SyncEvent(ProgressEvent):
    event_type: str  # "started", "complete", "failed", "canceled"
    timestamp: datetime
    model: str
    query_preview: str  # First ~30 chars of the query
    duration: float | None = None  # Duration in seconds
    error: str | None = None  # Error message if any


class AsyncEvent(ProgressEvent):
    request_id: int
    event_type: str  # "started", "complete", "failed", "canceled"
    timestamp: datetime
    model: str
    query_preview: str  # First ~30 chars of the query
    duration: float | None = None  # Duration in seconds
    error: str | None = None  # Error message if any


class ProgressHandler(Protocol):
    def handle_event(self, event: ProgressEvent) -> None:
        """Handle a progress event."""
        ...


class ProgressTracker:
    def __init__(self, handler: ProgressHandler):
        self.handler = handler

    def emit_event(self, event: ProgressEvent):
        self.handler.handle_event(event)


class ConcurrentSummaryEvent(ProgressEvent):
    """Event for concurrent operation summaries"""

    event_type: str  # "concurrent_start", "concurrent_complete"
    timestamp: datetime
    total: int
    successful: int = 0
    failed: int = 0
    duration: float | None = None


class ConcurrentTracker:
    """Tracks progress of concurrent operations with real-time updates"""

    def __init__(self, handler, total: int):
        self.handler = handler
        self.total = total
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()

    def emit_concurrent_start(self):
        """Emit start event for concurrent operations"""
        event = ConcurrentSummaryEvent(
            event_type="concurrent_start", timestamp=datetime.now(), total=self.total
        )
        if hasattr(self.handler, "handle_concurrent_start"):
            self.handler.handle_concurrent_start(self.total)
        else:
            # Fallback for handlers without concurrent support
            if hasattr(self.handler, "show_spinner"):
                self.handler.show_spinner(
                    "concurrent", f"Running {self.total} concurrent requests..."
                )
            else:
                self.handler.emit_started(
                    "concurrent", f"Starting: {self.total} concurrent requests"
                )

    def operation_started(self):
        """Called when an individual operation starts"""
        # For now, we don't need to track individual starts
        # Could be extended later for more detailed progress
        pass

    def operation_completed(self):
        """Called when an individual operation completes successfully"""
        self.completed += 1
        self._update_progress()

    def operation_failed(self):
        """Called when an individual operation fails"""
        self.failed += 1
        self._update_progress()

    def _update_progress(self):
        """Update live progress display"""
        if hasattr(self.handler, "update_concurrent_progress"):
            elapsed = time.time() - self.start_time
            running = self.total - self.completed - self.failed
            self.handler.update_concurrent_progress(
                completed=self.completed,
                total=self.total,
                running=running,
                failed=self.failed,
                elapsed=elapsed,
            )

    def emit_concurrent_complete(self):
        """Emit completion event for all concurrent operations"""
        duration = time.time() - self.start_time
        successful = self.completed

        if hasattr(self.handler, "handle_concurrent_complete"):
            self.handler.handle_concurrent_complete(successful, self.total, duration)
        else:
            # Fallback for handlers without concurrent support
            if hasattr(self.handler, "show_complete"):
                self.handler.show_complete(
                    "concurrent",
                    f"All requests complete: {successful}/{self.total} successful",
                    duration,
                )
            else:
                self.handler.emit_complete(
                    "concurrent",
                    f"All requests complete: {successful}/{self.total} successful",
                    duration,
                )
