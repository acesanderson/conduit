from __future__ import annotations
from conduit.apps.chat.engine.async_engine import ChatEngine
from conduit.apps.chat.ui.input_interface import InputInterface
from conduit.utils.progress.verbosity import Verbosity
import logging

logger = logging.getLogger(__name__)


class ChatApp:
    """
    The main chat application.
    """

    def __init__(
        self,
        engine: ChatEngine,
        input_interface: InputInterface,
        welcome_message: str = "",
        verbosity: Verbosity = Verbosity.SILENT,
    ):
        # Init variables
        self.engine: ChatEngine = engine
        self.input_interface: InputInterface = input_interface
        self.welcome_message: str = welcome_message
        self.verbosity: Verbosity = verbosity
        # Control variable for the main loop
        self.is_running: bool = True

    async def run(self) -> None:
        """
        Start the main event loop for the chat application.

            Initializes the session by clearing the screen and displaying the welcome
            message, then enters a continuous loop to process user input and generate
            responses until the application is terminated via the `is_running` flag.
        """
        # Clear the screen at the start of the chat session
        self.input_interface.clear_screen()
        if self.welcome_message:
            from rich.console import Console

            Console().print(self.welcome_message)

        while self.is_running:
            await self.run_once()

    async def run_once(self) -> None:
        """
        Execute a single cycle of the chat loop: get input, process it, and display output.

            Retrieves user input asynchronously. If the input starts with '/', it is routed
            to command execution; otherwise, it is processed as a conversation query via the
            engine. Handles input cancellation (e.g., Ctrl+C) by updating the running state
            and catches execution errors to prevent app crashes.
        """
        user_input = await self.input_interface.get_input()
        # Handle case where input can be cancelled (e.g., Ctrl+C)
        if user_input is None:
            self.is_running = False
            return

        try:
            if user_input.startswith("/"):
                output = await self.engine.execute_command(user_input, self)
                if output:
                    self.input_interface.show_message(output)
                # Commands may have side effects without returning output (e.g., /exit, /wipe)
            else:
                # If user only pressed enter, skip processing
                if not user_input.strip():
                    return
                output = await self.engine.handle_query(user_input)
                if output:
                    self.input_interface.show_message(output)
                else:
                    logger.warning(f"Query returned no output for: {user_input}")
        except Exception as e:
            logger.exception(f"Error processing input: {user_input}")
            self.input_interface.show_message(f"[red]Error: {e}[/red]")
