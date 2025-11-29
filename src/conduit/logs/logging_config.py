import logging
import os
from rich.logging import RichHandler


class PackagePathFilter(logging.Filter):
    """
    Prepends the root package name to record.pathname so Rich shows it in the right column.
    Example: my_app/utils/helpers.py -> my_app:helpers.py
    """

    def filter(self, record):
        record.root_package = record.name.split(".")[0]
        original_basename = os.path.basename(record.pathname)
        new_basename = f"{record.root_package}:{original_basename}"
        original_dirname = os.path.dirname(record.pathname)
        record.pathname = os.path.join(original_dirname, new_basename)
        return True


# --- Setup ---
log_level = int(os.getenv("PYTHON_LOG_LEVEL", "1"))  # Default to WARNING
levels = {1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}

rich_handler = RichHandler(rich_tracebacks=True, markup=True)
rich_handler.addFilter(PackagePathFilter())

logging.basicConfig(
    level=levels.get(log_level, logging.INFO),
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[rich_handler],
)
