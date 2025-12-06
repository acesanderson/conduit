"""
session odometer: in memory sqlite3 (conn = sqlite3.connect(":memory:")
conversation odomoter: saved in a sqlite3 file (similar naming convention to history store)
persistent odometer: saved in postgres
"""

from conduit.storage.odometer.token_event import TokenEvent
from pydantic import BaseModel, Field
from datetime import datetime


class Odometer(BaseModel):
    # Raw event storage
    events: list[TokenEvent] = Field(default_factory=list)

    # Aggregated totals
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)  # computed property or field

    # Provider-level aggregation
    provider_totals: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Structure: {"openai": {"input": 1000, "output": 500, "total": 1500}}

    # Model-level aggregation
    model_totals: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Structure: {"gpt-4o": {"input": 800, "output": 300, "total": 1100}}

    # Time-based aggregation
    daily_totals: dict[str, dict[str, int]] = Field(default_factory=dict)
    # Structure: {"2025-01-15": {"input": 500, "output": 200, "total": 700}}

    # Session metadata
    session_start: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    # Host tracking (for multi-machine scenarios)
    hosts: set[str] = Field(default_factory=set)

    def record(self, token_event: TokenEvent):
        """
        Add a TokenEvent to the odometer and update all aggregates.
        provider: str
        model: str
        input_tokens: int
        output_tokens: int
        timestamp: int
        host: str
        """
        self.events.append(token_event)
        self.total_input_tokens += token_event.input_tokens
        self.total_output_tokens += token_event.output_tokens
        self.total_tokens += token_event.input_tokens + token_event.output_tokens

        # Update provider totals
        if token_event.provider not in self.provider_totals:
            self.provider_totals[token_event.provider] = {
                "input": 0,
                "output": 0,
                "total": 0,
            }
        self.provider_totals[token_event.provider]["input"] += token_event.input_tokens
        self.provider_totals[token_event.provider]["output"] += (
            token_event.output_tokens
        )
        self.provider_totals[token_event.provider]["total"] += (
            token_event.input_tokens + token_event.output_tokens
        )

        # Update model totals
        if token_event.model not in self.model_totals:
            self.model_totals[token_event.model] = {"input": 0, "output": 0, "total": 0}
        self.model_totals[token_event.model]["input"] += token_event.input_tokens
        self.model_totals[token_event.model]["output"] += token_event.output_tokens
        self.model_totals[token_event.model]["total"] += (
            token_event.input_tokens + token_event.output_tokens
        )

        # Update daily totals
        date_str = datetime.fromtimestamp(token_event.timestamp).strftime("%Y-%m-%d")
        if date_str not in self.daily_totals:
            self.daily_totals[date_str] = {"input": 0, "output": 0, "total": 0}
        self.daily_totals[date_str]["input"] += token_event.input_tokens
        self.daily_totals[date_str]["output"] += token_event.output_tokens
        self.daily_totals[date_str]["total"] += (
            token_event.input_tokens + token_event.output_tokens
        )

        # Update hosts set
        self.hosts.add(token_event.host)

    # Query methods
    def get_provider_breakdown(self) -> dict[str, int]:
        # Return provider totals
        pass

    def get_model_breakdown(self) -> dict[str, int]:
        # Return model totals
        pass

    def get_daily_usage(self, date: str) -> dict[str, int]:
        # Return usage for specific day
        pass

    def get_recent_activity(self, hours: int = 24) -> list[TokenEvent]:
        # Filter events by timestamp
        pass

    def stats(self):
        """
        Pretty-print the stats from the odometer.
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
        table.add_row(
            "Total",
            str(self.total_input_tokens),
            str(self.total_output_tokens),
            str(self.total_tokens),
        )
        for provider, totals in self.provider_totals.items():
            table.add_row(
                Text(provider, style="blue"),
                str(totals["input"]),
                str(totals["output"]),
                str(totals["total"]),
            )
        for model, totals in self.model_totals.items():
            table.add_row(
                Text(model, style="blue"),
                str(totals["input"]),
                str(totals["output"]),
                str(totals["total"]),
            )
        for date, totals in self.daily_totals.items():
            table.add_row(
                Text(date, style="blue"),
                str(totals["input"]),
                str(totals["output"]),
                str(totals["total"]),
            )
        console.print(table)
