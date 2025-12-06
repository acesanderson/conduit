from conduit.storage.odometer.odometer_base import Odometer
from conduit.storage.odometer.session_odometer import SessionOdometer
from conduit.storage.odometer.database.persistence_backend import PersistenceBackend
from datetime import date, timedelta
from pydantic import Field


class PersistentOdometer(Odometer):
    """
    Handles long-term storage of odometer data.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Add this field declaration
    backend: PersistenceBackend | None = Field(default=None, exclude=True)

    def __init__(self, backend: PersistenceBackend | None = None, **kwargs):
        super().__init__(**kwargs)
        if backend is None:
            # Your existing backend creation logic
            try:
                from conduit.storage.odometer.database.pgres.postgres_backend import (
                    PostgresBackend,
                )

                self.backend = PostgresBackend()
            except Exception as e:
                print(f"Real error: {e}")
                raise
                # from conduit.storage.odometer.database.sqlite.SqliteBackend import SQLiteBackend
                #
                # self.backend = SQLiteBackend("odometer.db")
        else:
            self.backend = backend

    def sync_session_data(self, session_odometer: SessionOdometer):
        """
        Syncs the session odometer data to the persistent storage.
        """
        if session_odometer.events:
            self.backend.store_events(session_odometer.events)
            # Update local aggregates
            for event in session_odometer.events:
                self.record(event)

    def load_historical_data(self, days: int = 30):
        """Load recent historical data for analytics"""
        start_date = date.today() - timedelta(days=days)
        events = self.backend.get_events(start_date=start_date)

        # Rebuild aggregates from events
        self.clear_aggregates()
        for event in events:
            self.record(event)
