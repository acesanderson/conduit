from Chain.odometer.SessionOdometer import SessionOdometer
from Chain.odometer.ConversationOdometer import ConversationOdometer
from Chain.odometer.PersistentOdometer import PersistentOdometer
from Chain.odometer.TokenEvent import TokenEvent
import atexit, signal, sys


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
        self.session_odometer = SessionOdometer()
        self.conversation_odometers: dict[str, ConversationOdometer] = {}
        self.persistent_odometer = PersistentOdometer()
        # Register the persistent odometer to save on exit
        atexit.register(self._save_on_exit)
        # Register signal handlers for interrupts
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Termination

    def register_conversation_odometer(self, conversation_id: str):
        """
        Register a new ConversationOdometer for the given conversation ID.
        If an odometer already exists for the conversation, it will not be replaced.
        """
        if conversation_id not in self.conversation_odometers:
            self.conversation_odometers[conversation_id] = ConversationOdometer(
                conversation_id=conversation_id
            )

    def emit_token_event(self, event: TokenEvent):
        """
        Main entry point for the TokenEvents sent by Response.__init__().
        Distributes the event to the appropriate odometers.
        """
        self.session_odometer.record(event)
        self.persistent_odometer.record(event)
        if event.host in self.conversation_odometers:
            self.conversation_odometers[event.host].record(event)
        else:
            # If no conversation odometer exists for the host, create one
            self.register_conversation_odometer(event.host)
            self.conversation_odometers[event.host].record(event)

    def _signal_handler(self, signum, frame):
        """
        Handle interrupt signals
        """
        self._save_on_exit()
        sys.exit(0)

    def _save_on_exit(self):
        """
        Called on normal program exit.
        """
        try:
            self.persistent_odometer.sync_session_data(self.session_odometer)
            # Also sync any active conversation odometers
            for conv_odo in self.conversation_odometers.values():
                self.persistent_odometer.sync_session_data(self.session_odometer)
        except Exception as e:
            # Log error but don't crash on exit
            print(f"Warning: Failed to save odometer data on exit: {e}")
