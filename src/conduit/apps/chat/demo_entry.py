import asyncio
import logging
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.sync import Verbosity

# Assuming you placed demo_app.py in conduit/apps/chat/ alongside create_app.py
from conduit.apps.chat.demo_app import create_chat_app

# Configure logging to write to a file, NOT stdout,
# because stdout is owned by the TUI now.
logging.basicConfig(
    filename="conduit_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    # 1. Configuration
    options = ConduitOptions(project_name="demo-tui", verbosity=Verbosity.SILENT)

    # 2. Factory Creation (Force "demo" mode)
    app = create_chat_app(
        preferred_model="haiku",
        welcome_message="[bold cyan]Welcome to the TUI Demo![/bold cyan]\nType your message below and press [bold green]Enter[/bold green].",
        system_message="You are a helpful, concise assistant running in a TUI demo.",
        input_mode="demo",
        options=options,
    )

    # 3. Run the Event Loop
    try:
        await app.run()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully if it bubbles up
        pass


if __name__ == "__main__":
    asyncio.run(main())
