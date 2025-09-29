"""
Considerations:
- purpose of this is to abstract and simplify what I have in all my chat CLI apps (ask, tutorialize, leviathan, twig, cookbook)
- need to figure out how users can override the query class
- allow for grabbing stdin (so apps can be piped into)
- consider abstracting out the basic Conduit functionality as a mixin
- consider making a Command class instead of using functions like objects -- it's currently a little too clever and likely unreadable. Extensibility can be done by making Command classes. Maybe.

Usage:
- In its base form, the CLI class can be used to create a simple chat application, using Claude.
- You can extend it by adding your own arguments and functions.
- To add a function, create a method with the prefix "arg_" and decorate it with the cli_arg decorator. The abbreviation for the argument should be passed as an argument to the decorator (like @cli_arg("-m")).
"""

from conduit.message.messagestore import MessageStore
from conduit.model.model import Model
from conduit.prompt.prompt import Prompt
from conduit.conduit.sync_conduit import SyncConduit
from conduit.logs.logging_config import get_logger
from rich.console import Console
from utils import print_markdown
from inspect import signature
from typing import Callable
import sys, argparse

logger = get_logger(__name__)


def arg(abbreviation):
    """
    Decorator for adding arguments to the CLI.
    This is used to define the abbreviation for the argument.
    """

    def decorator(func):
        func.abbreviation = abbreviation
        return func

    return decorator


class CLI:
    """
    CLI class for Conduit module: use this to create command line-based chat applications.
    Define your arguments with the prefix "arg_" and decorate them with the @arg decorator.
    You can override the default method to change the default behavior. (def default(self):)
    """

    def __init__(
        self,
        name: str,
    ):
        """
        Initialize the CLI object with a name and optional history and log files.
        """
        self.name = name
        self.catalog = {}
        self.raw = False
        self.parser = self._init_parser()
        self.console = Console(width=120)

    def _print_markdown(self, markdown: str):
        print_markdown(markdown, console=self.console)

    def _init_parser(self):
        """
        Initialize the parser with all our arguments.
        """
        _parser = argparse.ArgumentParser(description=self.name)

        def catalog_arg(arg_func: Callable):
            """
            Take an argument function and add it to the parser.
            """
            arg_name = arg_func.__name__[4:]
            arg_doc = arg_func.__doc__
            param_count = len(signature(arg_func).parameters)
            abbreviation = getattr(arg_func, "abbreviation", "")

            # Default positional argument (empty abbreviation and takes parameter)
            if param_count == 1 and abbreviation == "":
                _parser.add_argument(arg_name, nargs="*", help=arg_doc)

            # Optional arguments with parameters (non-empty abbreviation and takes parameter)
            elif param_count == 1 and abbreviation != "" and abbreviation:
                _parser.add_argument(
                    abbreviation, type=str, nargs="?", dest=arg_name, help=arg_doc
                )

            # Flag arguments (no parameters beyond self)
            elif param_count == 0 and abbreviation != "" and abbreviation:
                _parser.add_argument(
                    abbreviation, action="store_true", dest=arg_name, help=arg_doc
                )
            else:
                print(
                    f"Warning: Skipping {arg_name} - unhandled case: params={param_count}, abbrev='{abbreviation}'"
                )

            return arg_name

        methods = [method for method in dir(self) if method.startswith("arg_")]
        for method in methods:
            method_name = catalog_arg(getattr(self, method))
            self.catalog[method_name] = getattr(self, method)
        return _parser

    def run(self):
        parsed_args = self.parser.parse_args()
        parsed_args = vars(parsed_args)
        # Detect null state
        is_null_state = all(not arg for arg in parsed_args.values())
        if is_null_state and hasattr(self, "_default"):
            self._default()
        else:
            # Not null, parse the arguments
            ## Start with the known "state setters"
            for arg in ["raw"]:
                if arg in parsed_args and parsed_args[arg]:
                    self.catalog[arg]()
            ## Then the remaining args
            for arg in parsed_args:
                if parsed_args[arg] and arg not in ["raw"]:
                    func = self.catalog[arg]
                    # Check if function expects parameters (remember, signature doesn't count 'self')
                    if (
                        len(signature(func).parameters) > 0
                    ):  # Has parameters beyond 'self'
                        func(parsed_args[arg])
                    else:
                        func()
            # TBD: the above is a hack that reflects the fact that some arguments are just flags, and should not be executed. One potential implementation would be to have TWO decorators; a flag and the arg decorator, where flags are not executed but rather accessed by the actual arg functions.
            sys.exit()

    def _default(self):
        """
        Default argument. This is the default function that runs if no other arguments are provided.
        Can be overridden by inheriting classes.
        """
        self.parser.print_help()
        sys.exit()


class ConduitCLI(CLI):
    def __init__(
        self,
        name: str = "Conduit CLI",
        history_file: str = "",
        log_file: str = "",
        pruning: bool = False,
    ):
        # Inherit super
        super().__init__(name=name)
        self.preferred_model = Model("claude")
        self.messagestore = MessageStore(
            console=self.console, history_file=history_file, pruning=True
        )
        SyncConduit._message_store = self.messagestore
        if history_file:
            self.messagestore.load()

    # Our arg methods
    @arg("")
    def arg_query(self, param):
        """
        Send a message.
        """
        # This is the default argument. Override in subclasses if needed.
        query = " ".join(param)
        prompt = Prompt(query)
        conduit = SyncConduit(prompt=prompt, model=self.preferred_model)
        response = conduit.run()
        if response.content:
            if self.raw:
                print(response)
            else:
                self._print_markdown(str(response.content))
        else:
            raise ValueError("No response found.")

    @arg("-hi")
    def arg_history(self):
        """
        Print the last 10 messages.
        """
        self.messagestore.view_history()

    @arg("-l")
    def arg_last(self):
        """
        Print the last message.
        """
        last_message = self.messagestore.last()
        if last_message:
            if self.raw:
                print(last_message.content)
            else:
                self._print_markdown(str(last_message.content))
        else:
            self.console.print("No messages yet.")

    @arg("-g")
    def arg_get(self, param):
        """
        Get a specific answer from the history.
        """
        if not param.isdigit():
            self.console.print("Please enter a valid number.")
            return
        retrieved_message = self.messagestore.get(int(param))
        if retrieved_message:
            try:
                if self.raw:
                    print(retrieved_message.content)
                else:
                    self._print_markdown(str(retrieved_message.content))
            except ValueError:
                self.console.print("Message not found.")
        else:
            self.console.print("Message not found.")

    @arg("-c")
    def arg_clear(self):
        """
        Clear the message history.
        """
        self.messagestore.clear()
        self.console.print("Message history cleared.")

    @arg("-m")
    def arg_model(self, param):
        """
        Specify a model.
        """
        self.preferred_model = Model(param)
        self.console.print(f"Model set to {param}.")

    @arg("-r")
    def arg_raw(self):
        """
        Print raw output.
        """
        self.raw = True


if __name__ == "__main__":
    c = ConduitCLI(name="Conduit Chat", history_file=".cli_history.log")
    c.run()
