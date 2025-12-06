from conduit.storage.odometer.session_odometer import SessionOdometer
from conduit.storage.odometer.persistent_odometer import PersistentOdometer
from conduit.storage.odometer.token_event import TokenEvent
import atexit
import signal
import sys


class OdometerRegistry:
    """
    A registry for odometers that allows for adding, removing, and retrieving odometers.
    Singleton which attaches to Model class as a class variable (._odometer_registry).

    Four purposes for this class:
    1. Centralized management of odometers for session, conversation, and persistent data.
    2. Handles the registration of new conversation odometers.
    3. Distributes TokenEvents to the appropriate odometers.
    4. Ensures that data is saved on program exit or interruption.
    """

    def __init__(self):
        """
        Initialize the odometer registry with an empty dictionary.
        """
        self.session_odometer: SessionOdometer = SessionOdometer()
        self.persistent_odometer: PersistentOdometer = PersistentOdometer()
        # Register the persistent odometer to save on exit
        _ = atexit.register(self._save_on_exit)
        # Register signal handlers for interrupts
        _ = signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
        _ = signal.signal(signal.SIGTERM, self._signal_handler)  # Termination
        self.is_saved: bool = False  # Idempotence guard

    def emit_token_event(self, event: TokenEvent) -> None:
        """
        Main entry point for the TokenEvents sent by Response.__init__().
        Distributes the event to the appropriate odometers.
        """
        self.session_odometer.record(event)
        self.persistent_odometer.record(event)

    def _signal_handler(self, signum, frame) -> None:
        """
        Handle interrupt signals
        """
        self._save_on_exit()
        sys.exit(0)

    def _save_on_exit(self) -> None:
        """
        Called on normal program exit.
        """
        if self.is_saved:
            return
        self.is_saved = True
        try:
            self.persistent_odometer.sync_session_data(self.session_odometer)
        except Exception as e:
            # Log error but don't crash on exit
            print(f"Warning: Failed to save odometer data on exit: {e}")
