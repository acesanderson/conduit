from __future__ import annotations
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
        schema = getattr(workflow, "schema", {})
        config_keys = set(self.config.keys())
        missing_hard = []

        for logical_name, details in schema.items():
            is_provided = any(k in config_keys for k in details["keys"])

            if not is_provided:
                # Failure Case 1: Defaults are disabled globally
                if not self.use_defaults:
                    missing_hard.append(f"{logical_name} (Defaults disabled)")
                # Failure Case 2: Defaults enabled, but this param has no default in code
                elif not details["has_code_default"]:
                    missing_hard.append(f"{logical_name} (No default in code)")

        # Infrastructure and detected keys
        all_valid_keys = {k for d in schema.values() for k in d["keys"]}
        all_valid_keys.update({"workflow_target", "entry_point"})
        unexpected = [k for k in config_keys if k not in all_valid_keys]

        if missing_hard or unexpected:
            self._print_diff(missing_hard, unexpected)

        if missing_hard:
            raise ConfigurationError(
                f"Workflow rejected due to non-compliant config: {missing_hard}"
            )

    def _print_diff(self, missing: list[str], unexpected: list[str]):
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        table = Table(box=None, show_header=False)

        if missing:
            table.add_row("[bold red]Missing (Required):[/]", ", ".join(missing))
        if unexpected:
            table.add_row("[bold yellow]Unused Config:[/]", ", ".join(unexpected))

        console.print(
            Panel(
                table,
                title="[bold]Workflow Configuration Validation[/]",
                border_style="red" if missing else "yellow",
            )
        )

    async def run(self, workflow: Workflow, *args, **kwargs) -> Any:
        # Mount context-level configuration policy
        token_defaults = context.use_defaults.set(self.use_defaults)
        token_conf = context.config.set(self.config)
        token_trace = context.trace.set(self.trace_log)
        token_discovery = context.discovery.set({})

        used_keys = set()
        token_access = context.access.set(used_keys)

        try:
            # Validate AFTER mounting context so resolve_param works if called during validation
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
