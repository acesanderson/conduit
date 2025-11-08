"""
Command registry and dispatcher for managing dynamically-registered chat commands.
This module provides the infrastructure for discovering, parsing, and executing chat commands
that follow a consistent interface. The `CommandRegistry` class serves as a centralized registry
that stores `Command` metadata objects and provides methods to look up commands by name, parse
command-line input into structured arguments, and execute them with proper error handling.

Commands are registered via the `@command` decorator (defined in `command.py`) and validated
to ensure function signatures match their declared parameter counts. The registry integrates
with the `ConduitChat` class to enable extensible command handling—new commands can be added
by simply defining decorated methods without modifying the chat loop itself. It handles both
parameterless commands (like `/help`) and parameterized commands (like `/set model gpt-4`)
by returning appropriate callables or partial functions for execution.

Usage Example:
```python
class ConduitChat(CommandDispatcher, CommandHandlers):
    pass

dispatcher = ConduitChat()
dispatcher.execute_command("/help")
dispatcher.execute_command('/model "claude-sonnet-4"')
```
"""

from conduit.chat.command import Command
from conduit.sync import Model, Verbosity, Response
from conduit.message.messagestore import MessageStore
from rich.console import RenderableType
import re
from instructor.exceptions import InstructorRetryException


class CommandDispatcher:
    """
    Registry and dispatcher for chat commands.
    Commands are dynamically registered via the @command decorator.
    LLM dependencies must be passed on init.
    """

    def __init__(
        self,
        model: str,
        message_store: MessageStore,
        system_message: str,
        verbosity: Verbosity,
    ) -> None:
        self.model: Model = Model(model)
        self.message_store: MessageStore = message_store
        self.system_message: str = system_message
        self.verbosity: Verbosity = verbosity
        # Dynamically registered commands
        self._commands: dict[str, Command] = {}
        self._register_commands()

    # Dynamically register commands decorated with @command
    def _register_commands(self) -> None:
        """
        Scan for decorated methods and register them.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_command"):
                cmd_template = attr._command
                # Create a new Command with the BOUND method
                cmd = Command(
                    name=cmd_template.name,
                    func=attr,
                    aliases=cmd_template.aliases,
                    param_count=cmd_template.param_count,
                )
                # Register primary name
                if cmd.name in self._commands:
                    raise ValueError(f"Duplicate command name: {cmd.name}")
                self._commands[cmd.name] = cmd
                # Register aliases
                for alias in cmd.aliases:
                    if alias in self._commands:
                        raise ValueError(f"Duplicate command/alias: {alias}")
                    self._commands[alias] = cmd

    def register_command(self, cmd: Command) -> None:
        """
        Manually register a command (alternative to decorator-based registration).
        """
        if cmd.name in self._commands:
            raise ValueError(f"Duplicate command name: {cmd.name}")
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            if alias in self._commands:
                raise ValueError(f"Duplicate command/alias: {alias}")
            self._commands[alias] = cmd

    # Discover and route commands
    def _split_args(self, text: str) -> list[str]:
        """
        Split arguments respecting quoted strings.
        shlex-like parsing.

        Examples:
            'arg1 arg2' → ['arg1', 'arg2']
            '"arg with spaces" arg2' → ['arg with spaces', 'arg2']
            'arg1 "arg 2" arg3' → ['arg1', 'arg 2', 'arg3']
        """
        pattern = r'"([^"]*)"|(\S+)'
        matches = re.findall(pattern, text)
        return [quoted or unquoted for quoted, unquoted in matches]

    def parse_command(self, text: str) -> tuple[Command, list[str] | None]:
        """
        Parse user input into command and arguments.
        Supports quoted strings for multi-word arguments.

        Examples:
            /model claude-sonnet-4
            /add "Python Essential Training" "SQL Basics"
            /search "machine learning" advanced

        Returns:
            (Command, None) for param_count=0
            (Command, [single_arg]) for param_count=1
            (Command, [arg1, arg2, ...]) for param_count="multi"

        Raises:
            ValueError: If not a command, unknown command, or invalid arguments
        """
        if not text.startswith("/"):
            raise ValueError("Not a command")

        # Remove leading '/' and split into parts
        command_text = text[1:]
        if not command_text:
            raise ValueError("Empty command")

        parts = self._split_args(command_text)

        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else None

        if cmd_name not in self._commands:
            raise ValueError(f"Unknown command: /{cmd_name}")

        cmd = self._commands[cmd_name]

        # Validate argument count
        if cmd.param_count == 0 and args:
            raise ValueError(f"Command /{cmd.name} does not accept parameters")
        elif cmd.param_count == 1 and (not args or len(args) != 1):
            raise ValueError(f"Command /{cmd.name} requires exactly one parameter")
        elif cmd.param_count == "multi" and not args:
            raise ValueError(f"Command /{cmd.name} requires at least one parameter")

        return cmd, args

    def execute_command(self, text: str) -> RenderableType | None:
        """
        Parse and execute a command.

        Raises:
            ValueError: If command parsing or execution fails
        """
        cmd, args = self.parse_command(text)
        output = cmd.execute(args)
        return output

    def get_all_commands(self) -> list[Command]:
        """
        Get a list of all registered commands (primary names only, no aliases).

        Returns:
            List of Command objects, sorted by name
        """
        seen = set()
        commands = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                commands.append(cmd)
        return sorted(commands, key=lambda c: c.name)

    # Our query command, directly accessed by the chat app
    def handle_query(self, user_input: str) -> RenderableType:
        """
        Send a query to the model and display the response.
        """
        # Ignore empty queries
        if user_input.strip() == "":
            return
        # Send query to model
        try:
            # Ensure system message is set
            if self.system_message:
                self.message_store.ensure_system_message(self.system_message)
            # Add user message to store
            self.message_store.add_new(role="user", content=user_input)

            # Query model with full message history
            response = self.model.query(
                self.message_store.messages, verbose=self.verbosity
            )

            assert isinstance(response, Response), "Expected Response from model query"

            # Add assistant response to store
            self.message_store.add_new(role="assistant", content=str(response))

            # Display response
            return str(response.content)

        except InstructorRetryException:
            # Network failure from instructor
            return "[red]Network error. Please try again.[/red]"

        except KeyboardInterrupt:
            # Allow canceling during model query
            print("\nQuery canceled.")
            raise
