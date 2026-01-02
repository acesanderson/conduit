from conduit.storage.odometer.odometer import Odometer
from conduit.storage.odometer.token_event import TokenEvent
from conduit.storage.odometer.pgres.postgres_backend_async import AsyncPostgresOdometer
from conduit.config import settings
import atexit
import signal
import sys
import json
import logging
from types import FrameType

logger = logging.getLogger(__name__)

RESCUE_FILE = settings.paths["DATA_DIR"] / "odometer_rescue.json"


class OdometerRegistry:
    """
    Central entry point for tracking token usage.

    Architecture:
    - Hot Path: flush() calls asyncpg to save to DB.
    - Cold Path: atexit/signal hooks dump to JSON file (Rescue File).
    - Recovery: On init, checks for Rescue File and ingests it.
    """

    def __init__(self):
        self.session_odometer: Odometer = Odometer()
        # We instantiate the backend here. Since it's lazy, it won't connect yet.
        self.async_backend = AsyncPostgresOdometer(db_name="conduit")
        self._saved_on_exit: bool = False

        # Register Hooks
        atexit.register(self._save_on_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def emit_token_event(self, event: TokenEvent) -> None:
        """Record an event in the in-memory buffer."""
        self.session_odometer.record(event)

    async def flush(self) -> None:
        """
        HOT PATH: Async flush to database.
        Call this periodically or at the end of requests.
        """
        # 1. Atomic Pop
        batch = self.session_odometer.pop_events()
        if not batch:
            return

        try:
            # 2. Async Write
            await self.async_backend.store_events(batch)
        except Exception as e:
            logger.error(f"Odometer async flush failed: {e}")
            # 3. Safety Requeue (so they are saved to disk if app crashes)
            self.session_odometer.requeue_events(batch)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle interrupt signals by triggering save_on_exit."""
        self._save_on_exit()
        sys.exit(0)

    def _save_on_exit(self) -> None:
        """
        COLD PATH: Synchronous dump to JSON file.
        Executed during interpreter shutdown.
        """
        if self._saved_on_exit:
            return
        self._saved_on_exit = True

        # Grab everything remaining
        batch = self.session_odometer.pop_events()
        if not batch:
            return

        try:
            # Simple JSON dump
            data = [e.model_dump() for e in batch]

            # If file exists (rare double crash), append
            if RESCUE_FILE.exists():
                try:
                    existing = json.loads(RESCUE_FILE.read_text())
                    if isinstance(existing, list):
                        data = existing + data
                except (json.JSONDecodeError, OSError):
                    pass  # Overwrite corrupt file

            RESCUE_FILE.parent.mkdir(parents=True, exist_ok=True)
            RESCUE_FILE.write_text(json.dumps(data))
            # Use print because logging might be dead
            print(f"[Odometer] Rescued {len(batch)} unsaved events to {RESCUE_FILE}")
        except Exception as e:
            print(f"[Odometer] CRITICAL: Failed to rescue events: {e}")

    async def recover(self) -> None:
        """
        Run this on server startup/middleware init to ingest rescued events.
        """
        if not RESCUE_FILE.exists():
            return

        logger.info(f"Odometer found rescue file: {RESCUE_FILE}")
        try:
            content = RESCUE_FILE.read_text()
            if not content:
                RESCUE_FILE.unlink()
                return

            data = json.loads(content)
            events = [TokenEvent(**item) for item in data]

            if events:
                logger.info(f"Recovering {len(events)} events from previous session...")
                await self.async_backend.store_events(events)

            # Clean up
            RESCUE_FILE.unlink()
            logger.info("Rescue file recovered and deleted.")

        except Exception as e:
            logger.error(f"Odometer recovery failed: {e}")
