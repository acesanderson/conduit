"""
ChatApp orchestrates the core event loop and message flow for an interactive multi-turn chat session, serving as the bridge between user input, command execution, and language model interactions. It manages a REPL that distinguishes between slash-prefixed commands (parsed and executed via CommandRegistry) and natural language queries (sent to the Model with full MessageStore history), enabling seamless context-aware conversations.

Usage:

```python
registry = CommandRegistry()
input_interface = BasicInput()
model = Model(...)
message_store = MessageStore()
app = ChatApp(
    registry=registry,
    input_interface=input_interface,
    welcome_message="Welcome to the ChatApp!",
)
app.run()
```
"""

from conduit.chat.engine import ConduitEngine
from conduit.chat.ui.input_interface import InputInterface


class ChatApp:
    """
    Interactive REPL for multi-turn chat sessions that bridges user input, command execution, and language model interactions.
    """

    def __init__(
        self,
        engine: ConduitEngine,
        input_interface: InputInterface,
        welcome_message: str,
    ):
        """
        Initialize all our dependencies.
        """
        self.engine: ConduitEngine = engine
        self.input_interface: InputInterface = input_interface
        self.welcome_message: str = welcome_message
        # Inject enginer into input interface if needed
        if hasattr(input_interface, "set_engine"):
            self.input_interface.set_engine(self.engine)

    def run(self) -> None:
        """
        Start the REPL loop.
        """
        self.input_interface.clear_screen()
        self.input_interface.show_message(self.welcome_message)

        while True:
            try:
                user_input = self.input_interface.get_input()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    try:
                        self._handle_command(user_input)
                    except KeyboardInterrupt:
                        self.input_interface.show_message(
                            "\nCommand canceled.", style="green"
                        )
                else:
                    # Handle chat query
                    if len(user_input.strip()) == 0:
                        continue
                    try:
                        response = self.engine.handle_query(user_input)
                        self.input_interface.show_message(response)
                    except KeyboardInterrupt:
                        self.input_interface.show_message(
                            "\nQuery canceled.", style="green"
                        )

            except ValueError as e:
                self.input_interface.show_message(str(e), style="red")

            except NotImplementedError:
                self.input_interface.show_message(
                    "Method not implemented yet.", style="red"
                )

            except KeyboardInterrupt:
                # Ctrl+C at the prompt itself -> exit app
                self.input_interface.show_message("\nGoodbye!", style="green")
                break

    def _handle_command(self, user_input: str) -> None:
        """
        Execute a command and display any output.
        """
        try:
            output = self.engine.execute_command(user_input)
            if output:
                self.input_interface.show_message(output)
        except SystemExit:
            self.input_interface.show_message("[bold cyan]Goodbye![/bold cyan]")
            raise
