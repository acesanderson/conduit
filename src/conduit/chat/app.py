"""
ChatApp is the main REPL controller for interactive chat sessions, bridging the command registry, language model, and persistent message store into a unified conversational interface.

The class manages the core event loop that reads user input, distinguishes between slash-prefixed commands and natural language queries, and routes each to the appropriate handler. Commands are delegated to a CommandRegistry for parsing and execution (allowing extensible command systems), while chat queries are sent to the Model with full conversation history maintained in MessageStore, enabling multi-turn dialogue with context awareness.

ChatApp integrates with Rich console for formatted output and handles graceful error recovery for network failures, canceled operations, and keyboard interrupts at both the individual-operation and application-exit levels. It's typically instantiated by a factory function that injects configured dependencies, making it suitable for use in CLI applications or as the core of chat-based agent systems.

Usage example:
```python
console = Console()
registry = CommandRegistry()
model = Model(...)
message_store = MessageStore()
app = ChatApp(
    registry=registry,
    model=model,
    console=console,
    message_store=message_store,
    welcome_message="Welcome to the ChatApp!",
    system_message="You are a helpful assistant."
)
app.run()
```
"""

from conduit.chat.registry import CommandRegistry
from conduit.model.model import Model
from conduit.progress.verbosity import Verbosity
from conduit.message.messagestore import MessageStore
from rich.console import Console
from rich.markdown import Markdown
from instructor.exceptions import InstructorRetryException


class ChatApp:
    """Main REPL application for chat interface."""

    def __init__(
        self,
        registry: CommandRegistry,
        model: Model,
        console: Console,
        message_store: MessageStore,
        welcome_message: str,
        system_message: str,
        verbosity: Verbosity,
    ):
        self.registry = registry
        self.model = model
        self.console = console
        self.message_store = message_store
        self.welcome_message = welcome_message
        self.system_message = system_message
        self.verbosity = verbosity

    def run(self) -> None:
        """Start the REPL loop."""
        self.console.clear()
        self.console.print(self.welcome_message)

        if self.system_message:
            self.message_store.ensure_system_message(self.system_message)

        while True:
            try:
                user_input = self.console.input("[bold gold3]>> [/bold gold3]")

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    try:
                        self._handle_command(user_input)
                    except KeyboardInterrupt:
                        self.console.print("\nCommand canceled.", style="green")
                else:
                    # Handle chat query
                    try:
                        self._handle_query(user_input)
                    except KeyboardInterrupt:
                        self.console.print("\nQuery canceled.", style="green")

            except ValueError as e:
                self.console.print(str(e), style="red")

            except NotImplementedError:
                self.console.print("Method not implemented yet.", style="red")

            except KeyboardInterrupt:
                # Ctrl+C at the prompt itself -> exit app
                self.console.print("\nGoodbye!", style="green")
                break

    def _handle_command(self, user_input: str) -> None:
        """Execute a command and display any output."""
        try:
            output = self.registry.execute_command(user_input)
            if output:
                self.console.print(output)
        except SystemExit:
            self.console.print("Goodbye!", style="green")
            raise

    def _handle_query(self, user_input: str) -> None:
        """Send a query to the model and display the response."""
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
            self.console.print(Markdown(str(response) + "\n"), style="blue")

        except InstructorRetryException:
            # Network failure from instructor
            self.console.print("Network error. Please try again.", style="red")

        except KeyboardInterrupt:
            # Allow canceling during model query
            self.console.print("\nQuery canceled.", style="green")
            raise  # Re-raise to be caught by outer loop
