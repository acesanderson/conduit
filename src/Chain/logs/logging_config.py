import logging, sys

# Global logger dictionary to ensure we return the same logger instance for each module
_loggers = {}

# In logging_config.py
_configured = False

def get_logger(name=None, level=None):
    """
    Returns the logger with the specified name, creating it if necessary.
    Using this function ensures the same logger instance is returned for the same name.

    Args:
        name: The logger name (typically __name__ from the calling module)
        level: Optional level to set for this specific logger

    Returns:
        A configured logger instance
    """
    global _loggers

    # Default to the root logger if no name is provided
    if name is None:
        name = "ChainLogger"

    if name in _loggers:
        logger = _loggers[name]
        if level is not None:
            logger.setLevel(level)
        return logger

    # Create a new logger
    logger = logging.getLogger(name)

    # Set level if provided; otherwise inherit from parent
    if level is not None:
        logger.setLevel(level)

    # Store in cache
    _loggers[name] = logger

    return logger

def configure_logging(
    level=logging.INFO,
    console=True,
):
    """
    Configure the root logging settings for the entire application.
    Call this once at application startup.

    Args:
        level: The logging level (default: INFO)
        log_file: Optional file to log to (default: None)
        trace_mode: If True, add detailed debugging info with line numbers
        console: If True, log to console (via stderr)
    """
    # Ensure this function is only called once
    global _configured
    if _configured:
        return get_logger("Chain")  # Return existing logger
    
    _configured = True

    # Create a log formatter with trace information if requested
    log_format = (
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d (%(funcName)s) - %(message)s"
    )

    formatter = logging.Formatter(log_format)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler if requested, using stderr instead of stdout
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Create the main application logger
    logger = get_logger("ChainLogger")
    logger.info(
        "Audio pipeline logging configured with level: %s, trace_mode: True",
        logging.getLevelName(level),
    )

    return logger
