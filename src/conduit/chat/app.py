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
    model=model,
    message_store=message_store,
    welcome_message="Welcome to the ChatApp!",
    system_message="You are a helpful assistant."
)
app.run()
```
"""

from conduit.chat.registry import CommandRegistry
from conduit.chat.ui.input_interface import InputInterface
from conduit.model.model import Model
from conduit.progress.verbosity import Verbosity
from conduit.message.messagestore import MessageStore
from rich.markdown import Markdown
from instructor.exceptions import InstructorRetryException


class ChatApp:
    """
    Interactive REPL for multi-turn chat sessions that bridges user input, command execution, and language model interactions.
    """

    def __init__(
        self,
        # REPL components
        registry: CommandRegistry,
        input_interface: InputInterface,
        # LLM configs
        model: Model,
        message_store: MessageStore,
        welcome_message: str,
        system_message: str,
        verbosity: Verbosity,
    ):
        # REPL components
        self.registry = registry
        self.input_interface = input_interface
        # LLM configs
        self.model = model
        self.message_store = message_store
        self.welcome_message = welcome_message
        self.system_message = system_message
        self.verbosity = verbosity

    def run(self) -> None:
        """
        Start the REPL loop.
        """
        self.input_interface.clear_screen()
        self.input_interface.show_message(self.welcome_message)

        if self.system_message:
            self.message_store.ensure_system_message(self.system_message)

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
                    try:
                        self._handle_query(user_input)
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
            output = self.registry.execute_command(user_input)
            if output:
                self.input_interface.show_message(output)
        except SystemExit:
            self.input_interface.show_message("[bold cyan]Goodbye![/bold cyan]")
            raise

    def _handle_query(self, user_input: str) -> None:
        """
        Send a query to the model and display the response.
        """
        try:
            # Add user message to store
            self.message_store.add_new(role="user", content=user_input)

            # Query model with full message history
            response = self.model.query(
                self.message_store.messages, verbose=self.verbosity
            )

            # Add assistant response to store
            self.message_store.add_new(role="assistant", content=str(response))

            # Display response
            self.input_interface.show_message(str(response.content))

        except InstructorRetryException:
            # Network failure from instructor
            self.input_interface.show_message(
                "Network error. Please try again.", style="red"
            )

        except KeyboardInterrupt:
            # Allow canceling during model query
            self.input_interface.show_message("\nQuery canceled.", style="green")
            raise  # Re-raise to be caught by outer loop
