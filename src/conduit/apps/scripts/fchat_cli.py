"""
fchat = "fancy chat".
This is Enhanced Input (prompt toolkit)
"""

from rich.console import Console
from conduit.sync import ConduitCache, Model, Verbosity
from conduit.domain.message.messagestore import MessageStore
from conduit.chat.create_app import create_chat_app
from conduit.chat.ui.enhanced_input import EnhancedInput
from xdg_base_dirs import xdg_config_home
import logging
import os

# Set up logging
log_level = int(os.getenv("PYTHON_LOG_LEVEL", "1"))
levels = {1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
logging.basicConfig(
    level=levels.get(log_level, logging.INFO), format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
INPUT_INTERFACE = EnhancedInput(console=Console())
PREFERRED_MODEL = "haiku"
WELCOME_MESSAGE = "[bold cyan]Conduit Chat. Type /exit to exit.[/bold cyan]"
MESSAGE_STORE = MessageStore()
SYSTEM_MESSAGE = (
    (xdg_config_home() / "conduit" / "system_message.jinja2").read_text()
    if (xdg_config_home() / "conduit" / "system_message.jinja2").exists()
    else ""
)
VERBOSITY = Verbosity.PROGRESS
# Attach our singletons
CACHE = ConduitCache(name="conduit")
Model.conduit_cache = CACHE
Model.console = Console()


def main():
    app = create_chat_app(
        input_interface=INPUT_INTERFACE,
        preferred_model=PREFERRED_MODEL,
        welcome_message=WELCOME_MESSAGE,
        system_message=SYSTEM_MESSAGE,
        message_store=MESSAGE_STORE,
        verbosity=VERBOSITY,
    )
    app.run()


if __name__ == "__main__":
    main()
