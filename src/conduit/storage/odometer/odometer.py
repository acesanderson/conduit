from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from conduit.storage.odometer.token_event import TokenEvent
import logging

logger = logging.getLogger(__name__)


class Odometer(BaseModel):
    """
    In-memory odometer for tracking token usage and derived aggregates.
    """

    # Raw event storage
    events: list[TokenEvent] = Field(default_factory=list)

    # Aggregated totals
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    # Provider-level aggregation
    provider_totals: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Model-level aggregation
    model_totals: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Time-based aggregation (by date string YYYY-MM-DD)
    daily_totals: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Session metadata
    session_start: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    # Host tracking
    hosts: set[str] = Field(default_factory=set)

    def record(self, token_event: TokenEvent) -> None:
        """
        Add a TokenEvent to the odometer and update all aggregates.
        """
        logger.debug(f"Recording TokenEvent: {token_event}")
        self.events.append(token_event)

        # Update Aggregates (Keep running totals in memory)
        self.total_input_tokens += token_event.input_tokens
        self.total_output_tokens += token_event.output_tokens
        self.total_tokens += token_event.input_tokens + token_event.output_tokens

        # Provider totals
        provider_key = str(token_event.provider)
        self._update_aggregate(self.provider_totals, provider_key, token_event)

        # Model totals
        self._update_aggregate(self.model_totals, token_event.model, token_event)

        # Daily totals
        date_str = datetime.fromtimestamp(token_event.timestamp).strftime("%Y-%m-%d")
        self._update_aggregate(self.daily_totals, date_str, token_event)

        self.hosts.add(token_event.host)
        self.last_updated = datetime.now()

    def _update_aggregate(self, target_dict: dict, key: str, event: TokenEvent):
        entry = target_dict.setdefault(key, {"input": 0, "output": 0, "total": 0})
        entry["input"] += event.input_tokens
        entry["output"] += event.output_tokens
        entry["total"] += event.input_tokens + event.output_tokens

    def pop_events(self) -> list[TokenEvent]:
        """
        Atomic retrieval: Return all current events and clear the internal buffer.
        Used by flusher/rescue to ensure no data is duplicated or left behind.
        """
        if not self.events:
            return []

        # Atomic swap
        current_batch = self.events
        self.events = []
        return current_batch

    def requeue_events(self, failed_events: list[TokenEvent]) -> None:
        """
        Safety valve: If a DB write fails, put the events back at the front of the queue
        so they can be rescued by the file dump on exit.
        """
        self.events = failed_events + self.events

    # Simple query helpers
    def get_provider_breakdown(self) -> dict[str, dict[str, int]]:
        """Return aggregate totals by provider."""
        return self.provider_totals

    def get_model_breakdown(self) -> dict[str, dict[str, int]]:
        """Return aggregate totals by model."""
        return self.model_totals

    def get_daily_usage(self, date_str: str) -> dict[str, int]:
        """
        Return usage totals for a specific day (YYYY-MM-DD).
        Missing dates return zeros.
        """
        return self.daily_totals.get(date_str, {"input": 0, "output": 0, "total": 0})

    def get_recent_activity(self, hours: int = 24) -> list[TokenEvent]:
        """
        Return events whose timestamps fall within the last `hours`.
        """
        if not self.events:
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_ts = int(cutoff.timestamp())
        return [e for e in self.events if (e.timestamp or 0) >= cutoff_ts]

    def clear(self) -> None:
        """
        Clear all events and aggregates.
        """
        self.events.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.provider_totals.clear()
        self.model_totals.clear()
        self.daily_totals.clear()
        self.hosts.clear()
        self.session_start = datetime.now()
        self.last_updated = self.session_start

    def stats(self) -> None:
        """
        Pretty-print the stats from the odometer using rich.
        """
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console()
        table = Table(title="Odometer Stats")
        table.add_column("Category", justify="left", style="cyan")
        table.add_column("Input Tokens", justify="right", style="green")
        table.add_column("Output Tokens", justify="right", style="yellow")
        table.add_column("Total Tokens", justify="right", style="magenta")

        # Overall totals
        table.add_row(
            "Total",
            str(self.total_input_tokens),
            str(self.total_output_tokens),
            str(self.total_tokens),
        )

        # Providers
        for provider, totals in self.provider_totals.items():
            table.add_row(
                Text(f"Provider: {provider}", style="blue"),
                str(totals["input"]),
                str(totals["output"]),
                str(totals["total"]),
            )

        # Models
        for model, totals in self.model_totals.items():
            table.add_row(
                Text(f"Model: {model}", style="blue"),
                str(totals["input"]),
                str(totals["output"]),
                str(totals["total"]),
            )

        # Daily breakdown
        for date_str, totals in self.daily_totals.items():
            table.add_row(
                Text(f"Date: {date_str}", style="blue"),
                str(totals["input"]),
                str(totals["output"]),
                str(totals["total"]),
            )

        console.print(table)
