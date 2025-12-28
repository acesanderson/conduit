import logging

from conduit.apps.chat.engine.async_engine import ChatEngine
from conduit.apps.chat.ui.input_interface import InputInterface

logger = logging.getLogger(__name__)


class ChatApp:
    """
    The main chat application.
    """

    def __init__(self, engine: ChatEngine, input_interface: InputInterface, welcome_message: str = "", verbosity: "Verbosity" = "SILENT"):
        self.engine = engine
        self.input_interface = input_interface
        self.is_running = True
        self.welcome_message = welcome_message
        self.verbosity = verbosity

    async def run(self) -> None:
        """
        Runs the main application loop.
        """
        if self.welcome_message:
            # Print welcome message before the prompt_toolkit session starts
            # to avoid duplicate display when the session initializes
            from rich.console import Console
            Console().print(self.welcome_message)

        while self.is_running:
            await self.run_once()

    async def run_once(self) -> None:
        """
        Runs one iteration of the chat loop.
        """
        user_input = await self.input_interface.get_input()
        if user_input is None:  # Handle case where input can be cancelled (e.g., Ctrl+C)
            self.is_running = False
            return

        try:
            if user_input.startswith("/"):
                output = await self.engine.execute_command(user_input, self)
                if output:
                    self.input_interface.show_message(output)
                # Commands may have side effects without returning output (e.g., /exit, /wipe)
            else:
                output = await self.engine.handle_query(user_input)
                if output:
                    self.input_interface.show_message(output)
                else:
                    logger.warning(f"Query returned no output for: {user_input}")
        except Exception as e:
            logger.exception(f"Error processing input: {user_input}")
            self.input_interface.show_message(f"[red]Error: {e}[/red]")





