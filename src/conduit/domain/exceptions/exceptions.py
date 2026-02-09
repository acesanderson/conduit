class ConduitError(Exception):
    """Base class for all conduit-related exceptions."""

    pass


class EngineError(ConduitError):
    """Base class for all engine-related exceptions."""

    pass


class ModelError(ConduitError):
    """Exception raised for errors related to model operations."""

    pass


class OrchestrationError(ConduitError):
    """Exception raised for errors related to conduit operations."""

    pass


class ConversationError(ConduitError):
    """Exception raised for errors related to conversation operations."""

    pass


class ToolError(ConduitError):
    """Exception raised for errors related to tool operations."""

    pass


class ToolExecutionError(ToolError):
    """Recoverable external failures - return to LLM as ToolMessage."""

    pass


class ToolConfigurationError(ToolError):
    """Application/config errors - fail fast."""

    pass


class ChatError(ConduitError):
    """Exception raised for errors related to chat operations."""

    pass


class CLIError(ConduitError):
    """Exception raised for errors related to CLI operations."""

    pass
