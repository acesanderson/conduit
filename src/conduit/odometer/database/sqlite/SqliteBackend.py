from conduit.odometer.database.PersistenceBackend import PersistenceBackend
from conduit.odometer.TokenEvent import TokenEvent
from typing import override
from datetime import date
from pathlib import Path

default_db_path = Path(__file__).parent / "odometer.db"


class SQLiteBackend(PersistenceBackend):
    def __init__(self, db_path: str | Path = default_db_path):
        self.db_path: str | Path = db_path
        raise NotImplementedError(
            "SQLite backend is not yet implemented. Please use PostgreSQL backend."
        )

    @override
    def store_events(self, events: list[TokenEvent]) -> None:
        # Local SQLite storage
        pass

    @override
    def get_events(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        provider: str | None = None,
        model: str | None = None,
        host: str | None = None,
    ) -> list[TokenEvent]:
        # Query with filters
        return []

    @override
    def get_aggregates(
        self,
        group_by: str,  # "provider", "model", "host", "date"
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, dict[str, int]]:
        # Aggregation queries
        return {}
