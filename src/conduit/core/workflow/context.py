from dataclasses import dataclass, field
from contextvars import ContextVar


@dataclass(frozen=True)
class WorkflowContext:
    """
    Singleton registry for all Workflow-related ContextVars.
    Access these via the exported 'context' instance.
    """

    # Configuration (Read-Only for steps)
    config: ContextVar[dict] = field(
        default_factory=lambda: ContextVar("config", default={})
    )

    # Trace Log (Append-Only)
    trace: ContextVar[list | None] = field(
        default_factory=lambda: ContextVar("trace", default=None)
    )

    # Discovery Log (Write-Only, for schema generation)
    discovery: ContextVar[dict | None] = field(
        default_factory=lambda: ContextVar("config_discovery", default=None)
    )

    # Access Log (Write-Only, for drift detection)
    access: ContextVar[set | None] = field(
        default_factory=lambda: ContextVar("config_access", default=None)
    )

    # Step Metadata (Read/Write, scratchpad for the active step)
    step_meta: ContextVar[dict | None] = field(
        default_factory=lambda: ContextVar("step_meta", default=None)
    )


# The Singleton Instance
# Import THIS object in your other files.
context = WorkflowContext()
