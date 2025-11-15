class ConduitChatError(Exception):
    """Base exception for all application-specific errors."""

    pass


class CommandError(ConduitChatError):
    """Error related to command parsing or execution."""

    pass
