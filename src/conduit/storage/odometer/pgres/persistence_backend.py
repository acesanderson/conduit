from abc import ABC, abstractmethod
from datetime import date

from conduit.storage.odometer.token_event import TokenEvent


class PersistenceBackend(ABC):
    """Abstract interface for odometer persistence."""

    @abstractmethod
    def store_events(self, events: list[TokenEvent]) -> None:
        """Store raw token events."""
        raise NotImplementedError

    @abstractmethod
    def get_events(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        provider: str | None = None,
        model: str | None = None,
        host: str | None = None,
    ) -> list[TokenEvent]:
        """Query token events with filters."""
        raise NotImplementedError

    @abstractmethod
    def get_aggregates(
        self,
        group_by: str,  # "provider", "model", "host", "date"
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get aggregated statistics by group key."""
        raise NotImplementedError
