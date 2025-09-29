"""
Extensible chat class for CLI chat applications.
Add more commands by extending the commands list + defining a "command_" method.

TODO:
- more commands
    - [x] dynamic registration of new command methods
    - allow getting a message from history allow branching from somewhere in history allow pruning of history
    - allow saving history, saving a message
    - allow capturing screenshots for vision-based queries
- Leviathan extension
    - create a Chat app in Leviathan that incorporates ask, tutorialize, cookbook, etc.
- Mentor extension
    - system message
    - RAG: cosmo data from postgres, similarity search from chroma, direct access of courses from mongod
    - Curation objects: create, view, edit
    - blacklist course (for session)
- Learning from the Mentor implementation for more Agentic use cases
    - prompts
    - resources
    - tools
"""

from conduit.conduit.sync_conduit import SyncConduit
from conduit.model.model import Model
from conduit.message.messagestore import MessageStore
from conduit.message.textmessage import TextMessage
from conduit.message.message import Message
from conduit.message.messages import Messages
from conduit.cache.cache import ConduitCache
from conduit.logs.logging_config import get_logger
from rich.console import Console
from rich.markdown import Markdown
from instructor.exceptions import InstructorRetryException
from functools import partial
from typing import Callable, Optional
from pathlib import Path
import sys, inspect, readline  # Enables completion in the console


class Chat:
    """
    Basic CLI chat implementation.
    """

    def __init__(
        self,
        model: Model = Model("claude-3-5-haiku-20241022"),
        messagestore: Optional[MessageStore] = SyncConduit._message_store,
        console: Optional[Console] = SyncConduit._console,
    ):
        """
        User can inject their own messagestore, console, and model, otherwise defaults are used.
        """
        self.model = model
        if not console:
            self.console = Console(width=120)
        else:
            self.console = console
        self.messagestore = messagestore
        self.welcome_message = "[green]Hello! Type /exit to exit.[/green]"
        self.system_message: TextMessage | None = None
        self.commands = self.get_commands()
        self.log_file: str | Path = ""  # Off by default, but can be initialized.

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

    # Command methods
    def command_exit(self):
        """
        Exit the chat.
        """
        self.console.print("Goodbye!", style="green")
        exit()

    def command_help(self):
        """
        Display the help message.
        """
        commands = sorted(self.get_commands())
        help_message = "Commands:\n"
        for command in commands:
            command_name = command.replace("command_", "").replace("_", " ")
            command_func = getattr(self, command)
            try:
                command_docs = command_func.__doc__.strip()
            except AttributeError:
                print(f"Command {command_name} is missing a docstring.")
                sys.exit()
            help_message += (
                f"/[purple]{command_name}[/purple]: [green]{command_docs}[/green]\n"
            )
        self.console.print(help_message)

    def command_clear_history(self):
        """
        Clear the message history.
        """
        if self.messagestore:
            self.messagestore.clear()
            self.console.print("Message history cleared.", style="green")
        else:
            self.console.print("No message store available.", style="red")

    def command_clear(self):
        """
        Clear the screen.
        """
        self.console.clear()

    def command_show_history(self):
        """
        Display the chat history.
        """
        if self.messagestore:
            self.messagestore.view_history()

    def command_show_models(self):
        """
        Display available models.
        """
        self.console.print(Model.models, style="green")

    def command_show_model(self):
        """
        Display the current model.
        """
        self.console.print(f"Current model: {self.model.model}", style="green")

    def command_set_model(self, param: str):
        """
        Set the current model.
        """
        try:
            self.model = Model(param)
            self.console.print(f"Set model to {param}", style="green")
        except ValueError:
            self.console.print(f"Invalid model: {param}", style="red")

    def command_paste_image(self):
        """
        Use this when you have an image in clipboard that you want to submit as context for LLM.
        This gets saved as self.clipboard_image.
        """
        from Conduit.message.imagemessage import ImageMessage
        import os

        if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
            self.console.print("Image paste not available over SSH.", style="red")
            return

        import warnings
        from PIL import ImageGrab
        import base64, io

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress PIL warnings
            image = ImageGrab.grabclipboard()

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            # Save for next query
            self.console.print("Image captured! What is your query?", style="green")
            user_input = self.console.input(
                "[bold green]Query about image[/bold green]: "
            )
            # Build our ImageMessage
            text_content = user_input
            image_content = img_base64
            mime_type = "image/png"
            role = "user"
            imagemessage = ImageMessage(
                role=role,
                text_content=text_content,
                image_content=image_content,
                mime_type=mime_type,
            )
            self.messagestore.append(imagemessage)
            response = self.query_model([self.messagestore.last()])
            self.console.print(response)

        else:
            self.console.print("No image detected.", style="red")

    def command_wipe_image(self):
        """
        Delete image from memory.
        """
        if self.clipboard_image:
            self.clipboard_image = None
            self.console.print("Image deleted.", style="green")
        else:
            self.console.print("No image to delete.", style="red")

    # Main query method
    def query_model(self, query_input: list[Message] | Messages) -> str | None:
        """
        Takes either a string or a list of Message objects.
        """
        response = str(self.model.query(query_input, verbose="v"))
        if self.messagestore:
            self.messagestore.add_new(role="assistant", content=str(response))
        return response

    # Main chat loop
    def chat(self):
        self.console.clear()
        self.console.print(self.welcome_message)
        # If user passed a messagestore already, use that, otherwise declare a new one.
        if self.system_message:
            self.messagestore.append(self.system_message)
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
                            if self.messagestore:
                                self.messagestore.add_new(
                                    role="user", content=user_input
                                )
                                response = self.query_model(self.messagestore.messages)
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
        except KeyboardInterrupt:
            # User can exit the chat with Ctrl+C
            self.console.print("\nGoodbye!", style="green")
