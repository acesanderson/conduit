"""
Extensible chat class for CLI chat applications.
Add more commands by extending the commands list + defining a "command_" method, in the Handlers mixin.
(Or create a new mixin, and mix it in)

TODO:
- more commands
    - [x] dynamic registration of new command methods
    - allow getting a message from history allow branching from somewhere in history
    - allow pruning of history
    - allow saving history, saving a message
    - allow capturing screenshots for vision-based queries (classes stubbed out for this)
"""

from conduit.model.model import Model
from conduit.message.messagestore import MessageStore
from conduit.message.textmessage import TextMessage
from conduit.message.message import Message
from conduit.message.messages import Messages
from conduit.chat.handlers import Handlers
from rich.console import Console
from rich.markdown import Markdown
from instructor.exceptions import InstructorRetryException
from functools import partial
from collections.abc import Callable
import inspect
import logging

# Handlers mixin needs these
import sys
import os
import readline

logger = logging.getLogger(__name__)


class BaseChat:
    """
    Basic CLI chat implementation.
    Will not run without handlers mixed in (see ConduitChat definition at bottom of file).
    """

    def __init__(
        self,
        preferred_model: str,
        welcome_message: str,
        system_message: str,
        message_store: MessageStore,
        console: Console,
    ):
        # Inputs
        self.preferred_model = preferred_model
        self.welcome_message = welcome_message
        self.system_message = system_message
        if system_message == "":
            logger.warning("No system message initiated. Is this intentional?")
        self.message_store = message_store
        self.console = console
        # Dynamically generated attrs
        self.model = Model(preferred_model)
        self.commands = self.get_commands()

    def parse_input(self, input: str) -> Callable | partial | None:
        """
        Commands start with a slash. This method parses the input and returns the corresponding method.
        If command takes a param, this returns a partial function.
        If command is not found, it returns None (and the chat loop will handle it).
        """
        commands = self.get_commands()

        # Not a command; return None.
        if not input.startswith("/"):
            return None

        # Sort commands by length, so, for example, "show models" is invoked before "show model".
        commands = sorted(commands, key=len, reverse=True)

        # Parse for the type of command; this also involves catching parameters.
        for command in commands:
            command_string = command.replace("command_", "").replace("_", " ")
            if input.startswith("/" + command_string):
                # Check if the command has parameters
                sig = inspect.signature(getattr(self, command))
                if sig.parameters:
                    parametrized = True
                else:
                    parametrized = False
                # Check if input has parameters
                param = input[len(command_string) + 2 :]
                # Conditional return
                if param and parametrized:
                    return partial(getattr(self, command), param)
                elif param and not parametrized:
                    raise ValueError("Command does not take parameters.")
                elif not param and parametrized:
                    raise ValueError("Command requires parameters.")
                else:
                    return getattr(self, command)
        # Command not found
        raise ValueError("Command not found.")

    def get_commands(self) -> list[str]:
        """
        Dynamic inventory of "command_" methods.
        If you extend this with more methods, make sure they follow the "command_" naming convention.
        """
        commands = [attr for attr in dir(self) if attr.startswith("command_")]
        return commands

    # Main query method
    def query_model(self, query_input: list[Message] | Messages) -> str | None:
        """
        Takes either a string or a list of Message objects.
        """
        response = str(self.model.query(query_input, verbose="v"))
        if self.message_store:
            self.message_store.add_new(role="assistant", content=str(response))
        return response

    # Main chat loop
    def chat(self):
        self.console.clear()
        self.console.print(self.welcome_message)
        if self.system_message:
            self.message_store.ensure_system_message(self.system_message)
        try:
            while True:
                try:
                    user_input = self.console.input("[bold gold3]>> [/bold gold3]")
                    # Capture empty input
                    if not user_input:
                        continue
                    # Process commands
                    if user_input.startswith("/"):
                        command = self.parse_input(user_input)
                        if callable(command):
                            try:
                                command()
                                continue
                            except KeyboardInterrupt:
                                # User can cancel commands with Ctrl+C
                                self.console.print("\nCommand canceled.", style="green")
                                continue
                        else:
                            self.console.print("Invalid command.", style="red")
                            continue
                    else:
                        # Process query
                        try:
                            if self.message_store:
                                self.message_store.add_new(
                                    role="user", content=user_input
                                )
                                response = self.query_model(self.message_store.messages)
                            else:
                                response = self.query_model(
                                    [TextMessage(role="user", content=user_input)]
                                )
                            self.console.print(
                                Markdown(str(response) + "\n"), style="blue"
                            )
                            continue
                        except KeyboardInterrupt:
                            # User can cancel query with Ctrl+C
                            self.console.print("\nQuery canceled.", style="green")
                            continue
                        except InstructorRetryException:
                            # This exception is raised if there is some network failure from instructor.
                            self.console.print(
                                "Network error. Please try again.", style="red"
                            )
                except ValueError as e:
                    # If command not found, or commands throw an error, catch it and continue.
                    self.console.print(str(e), style="red")
                    continue
                except NotImplementedError:
                    self.console.print("Method not implemented yet.", style="red")
                    continue
        except KeyboardInterrupt:
            # User can exit the chat with Ctrl+C
            self.console.print("\nGoodbye!", style="green")


class ConduitChat(BaseChat, Handlers): ...
