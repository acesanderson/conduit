"""
fchat = "fancy chat".
This script launches the chat application with the enhanced (prompt_toolkit) input interface.
"""

import asyncio
import logging
import os

from rich.console import Console

from conduit.config import settings
from conduit.apps.chat.create_app import create_chat_app
from conduit.domain.config.conduit_options import ConduitOptions

# Set up logging
log_level = int(os.getenv("PYTHON_LOG_LEVEL", "1"))
levels = {1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
logging.basicConfig(
    level=levels.get(log_level, logging.INFO), format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CONSOLE = Console()
PREFERRED_MODEL = settings.preferred_model
WELCOME_MESSAGE = "[bold cyan]Conduit Chat (Enhanced). Type /exit to exit.[/bold cyan]"
SYSTEM_MESSAGE = settings.system_prompt
VERBOSITY = settings.default_verbosity

OPTIONS = ConduitOptions(
    project_name="conduit-fchat",
    verbosity=VERBOSITY,
    console=CONSOLE,
    cache=settings.default_cache("conduit-fchat"),
)

async def async_main():
    """
    Initializes and runs the asynchronous chat application.
    """
    app = create_chat_app(
        input_mode="enhanced",
        preferred_model=PREFERRED_MODEL,
        welcome_message=WELCOME_MESSAGE,
        system_message=SYSTEM_MESSAGE,
        options=OPTIONS,
    )
    
    await app.run()

def main():
    """
    Synchronous entry point to run the async main function.
    """
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()