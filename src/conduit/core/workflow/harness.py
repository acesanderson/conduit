from __future__ import annotations
import sys
from typing import Any, TYPE_CHECKING
from conduit.core.workflow.context import context

if TYPE_CHECKING:
    from conduit.core.workflow.protocols import Workflow


class ConfigurationError(Exception):
    """Raised when the workflow configuration does not meet the Strategy requirements."""

    pass


class ConduitHarness:
    def __init__(self, config: dict = None, use_defaults: bool = False):
        self.config = config or {}
        self.use_defaults = use_defaults
        self.trace_log: list = []
        self._discovered_config: dict[str, Any] = {}
        self._final_used_keys: set[str] = set()

    def validate_config(self, workflow: Workflow):
        """
        Validates config against the workflow schema.
        Exits process with a formatted diff if non-compliant.
        """
        # Defensive check: ensure schema is at least an empty dict
        schema = getattr(workflow, "schema", {}) or {}
        config_keys = set(self.config.keys())
        missing_hard = []

        for logical_name, details in schema.items():
            # 'details["keys"]' contains [scoped_key, flat_key]
            # e.g., ["OneShotSummarizer.model", "model"]
            is_provided = any(k in config_keys for k in details["keys"])

            if not is_provided:
                if not self.use_defaults:
                    missing_hard.append(logical_name)
                elif not details.get("has_code_default", False):
                    missing_hard.append(f"{logical_name} (No default in code)")

        # Determine unexpected keys (keys in config that aren't in any step's schema)
        all_valid_keys = {k for d in schema.values() for k in d["keys"]}
        all_valid_keys.update({"workflow_target", "entry_point"})

        unexpected = [k for k in config_keys if k not in all_valid_keys]

        if missing_hard or unexpected:
            self._print_diff(missing_hard, unexpected)

        if missing_hard:
            # Hard exit on missing required parameters
            sys.exit(1)

    def _print_diff(self, missing: list[str], unexpected: list[str]):
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        # We use a table for structural layout inside the panel
        layout_table = Table.grid(padding=(0, 1))

        if missing:
            layout_table.add_row(
                Text("Config dict is missing required params:", style="bold red")
            )
            # Add each missing param as a bullet point
            for param in missing:
                layout_table.add_row(f"  • [cyan]{param}[/cyan]")

            if not self.use_defaults:
                layout_table.add_row(
                    Text(
                        "(Note: use_defaults=False; code-level fallbacks ignored)",
                        style="dim",
                    )
                )

        if unexpected:
            if missing:
                layout_table.add_row("")  # Spacer
            layout_table.add_row(
                Text("Config dict has unexpected params:", style="bold yellow")
            )
            for param in unexpected:
                layout_table.add_row(f"  • [yellow]{param}[/yellow]")

        console.print("\n")
        console.print(
            Panel(
                layout_table,
                title="[bold red]Workflow Configuration Non-Compliance[/]"
                if missing
                else "[bold yellow]Workflow Configuration Warning[/]",
                border_style="red" if missing else "yellow",
                expand=False,
                padding=(1, 2),
            )
        )

    async def run(self, workflow: Workflow, *args, **kwargs) -> Any:
        token_defaults = context.use_defaults.set(self.use_defaults)
        token_conf = context.config.set(self.config)
        token_trace = context.trace.set(self.trace_log)
        token_discovery = context.discovery.set({})

        used_keys = set()
        token_access = context.access.set(used_keys)

        try:
            self.validate_config(workflow)
            return await workflow(*args, **kwargs)
        finally:
            discovery_snapshot = context.discovery.get()
            if discovery_snapshot:
                self._discovered_config = discovery_snapshot.copy()
            self._final_used_keys = used_keys.copy()

            context.use_defaults.reset(token_defaults)
            context.config.reset(token_conf)
            context.trace.reset(token_trace)
            context.discovery.reset(token_discovery)
            context.access.reset(token_access)

    def report_available_config(self) -> dict:
        report = {"global": {}, "scoped": {}}
        for full_key, default_val in self._discovered_config.items():
            if "." in full_key:
                scope, key = full_key.split(".", 1)
                report["scoped"].setdefault(scope, {})[key] = default_val
            else:
                report["global"][full_key] = default_val
        return report

    def report_unused_config(self) -> list[str]:
        accessed_keys = self._final_used_keys
        unused = [k for k in self.config.keys() if k not in accessed_keys]
        unused.sort()
        return unused

    def view_trace(self):
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Step")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Status")
        table.add_column("Metadata")
        table.add_column("Output", overflow="fold")

        for entry in self.trace_log:
            output = (
                str(entry["output"])[:47] + "..."
                if len(str(entry["output"])) > 50
                else str(entry["output"])
            )
            meta_str = ""
            if entry.get("metadata"):
                if "config_resolutions" in entry["metadata"]:
                    entry["metadata"]["configs_resolved"] = len(
                        entry["metadata"].pop("config_resolutions")
                    )
                meta_str = ", ".join(f"{k}: {v}" for k, v in entry["metadata"].items())

            table.add_row(
                entry["step"], str(entry["duration"]), entry["status"], meta_str, output
            )
        console.print(table)

    @property
    def trace(self) -> list:
        """Exposes the trace log for the current execution."""
        return self.trace_log
