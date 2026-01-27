from typing import Any
from conduit.core.workflow.context import context
from conduit.core.workflow.protocols import Workflow
import ast
import inspect
from collections import deque


class ConduitHarness:
    """
    The runtime container that manages Observability and Configuration contexts.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.trace_log: list = []
        # These are populated after a run()
        self._discovered_config: dict[str, Any] = {}
        self._final_used_keys: set[str] = set()

    async def run(self, workflow: Workflow, *args, **kwargs) -> Any:
        """
        Executes a workflow within the managed context.
        """
        # Mount Context
        token_conf = context.config.set(self.config)
        token_trace = context.trace.set(self.trace_log)

        # Initialize logs
        token_discovery = context.discovery.set({})
        # Track accessed keys for this run
        used_keys = set()
        token_access = context.access.set(used_keys)

        try:
            return await workflow(*args, **kwargs)
        finally:
            # Capture state before reset
            discovery_snapshot = context.discovery.get()
            if discovery_snapshot:
                self._discovered_config = discovery_snapshot.copy()

            self._final_used_keys = used_keys.copy()

            # Unmount Context
            context.config.reset(token_conf)
            context.trace.reset(token_trace)
            context.discovery.reset(token_discovery)
            context.access.reset(token_access)

    @property
    def trace(self) -> list:
        return self.trace_log

    def report_available_config(self) -> dict:
        """
        Returns a structured dictionary of all configuration options
        that were requested during the workflow execution.

        Returns:
            {
                "global": { "key": default_value },
                "scoped": { "ScopeName": { "key": default_value } }
            }
        """
        report = {"global": {}, "scoped": {}}

        for full_key, default_val in self._discovered_config.items():
            if "." in full_key:
                scope, key = full_key.split(".", 1)
                if scope not in report["scoped"]:
                    report["scoped"][scope] = {}
                report["scoped"][scope][key] = default_val
            else:
                report["global"][full_key] = default_val

        return report

    def report_unused_config(self) -> list[str]:
        """
        Returns a list of keys present in the harness config that were
        NEVER accessed by the workflow.

        This works by comparing the input config keys against the
        accessed keys tracked during execution.
        """
        unused = []

        # Keys that were actually read by resolve_param
        accessed_keys = self._final_used_keys

        for user_key in self.config.keys():
            if user_key not in accessed_keys:
                unused.append(user_key)

        unused.sort()
        return unused

    def view_trace(self):
        """
        Simple pretty-printer for the trace log using Rich.
        """
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
            output = str(entry["output"])
            if len(output) > 50:
                output = output[:47] + "..."

            # Format metadata nicely
            meta_str = ""
            if entry.get("metadata"):
                # Simplify config resolution display
                if "config_resolutions" in entry["metadata"]:
                    resolutions = entry["metadata"].pop("config_resolutions")
                    entry["metadata"]["configs_resolved"] = len(resolutions)

                meta_str = ", ".join(f"{k}: {v}" for k, v in entry["metadata"].items())

            table.add_row(
                entry["step"],
                str(entry["duration"]),
                entry["status"],
                meta_str,
                output,
            )

        console.print(table)
