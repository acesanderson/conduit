"""
In-memory buffer for token usage events.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from conduit.storage.odometer.token_event import TokenEvent


class Odometer:
    """
    Thread-safe in-memory buffer for token usage events.
    """

    def __init__(self):
        self._events: list[TokenEvent] = []
        self._lock = threading.Lock()

    def record(self, event: TokenEvent) -> None:
        """Record a token usage event."""
        with self._lock:
            self._events.append(event)

    def pop_events(self) -> list[TokenEvent]:
        """Atomically pop all events from the buffer."""
        with self._lock:
            events = self._events.copy()
            self._events.clear()
            return events

    def peek_events(self) -> list[TokenEvent]:
        """Get a copy of all events without clearing the buffer."""
        with self._lock:
            return self._events.copy()

    def clear(self) -> None:
        """Clear all events from the buffer."""
        with self._lock:
            self._events.clear()

    def count(self) -> int:
        """Get the number of events in the buffer."""
        with self._lock:
            return len(self._events)
