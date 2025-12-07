from conduit.storage.odometer.pgres.postgres_backend import PostgresBackend
from conduit.storage.odometer.odometer import Odometer
from conduit.storage.odometer.token_event import TokenEvent
import atexit
import signal
import sys
from types import FrameType
import logging

logger = logging.getLogger(__name__)


class OdometerRegistry:
    """
    Central entry point for tracking token usage in-memory and flushing to persistence.

    This is intended to be attached as a singleton on the Model class
    (e.g. Model._odometer_registry).
    """

    def __init__(self):
        self.session_odometer: Odometer = Odometer()
        self.backend: PostgresBackend = PostgresBackend()
        self._saved: bool = False  # idempotence guard

        _ = atexit.register(self._save_on_exit)
        _ = signal.signal(signal.SIGINT, self._signal_handler)
        _ = signal.signal(signal.SIGTERM, self._signal_handler)

    def emit_token_event(self, event: TokenEvent) -> None:
        """
        Main entry point for TokenEvents sent by Response.__init__().
        Updates the in-memory odometer; persistence happens on exit.
        """
        logger.debug(f"OdometerRegistry received TokenEvent: {event}")
        self.session_odometer.record(event)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """
        Handle interrupt signals: flush once, then exit.
        """
        self._save_on_exit()
        sys.exit(0)

    def _save_on_exit(self) -> None:
        """
        Called on normal program exit and from the signal handler.
        Idempotent: multiple calls are safe.
        """
        logger.debug("OdometerRegistry saving odometer data on exit...")
        if self._saved:
            return

        self._saved = True
        try:
            if self.session_odometer.events:
                self.backend.store_events(self.session_odometer.events)
        except Exception as e:
            # Log error but don't crash on exit
            print(f"Warning: Failed to save odometer data on exit: {e}")
