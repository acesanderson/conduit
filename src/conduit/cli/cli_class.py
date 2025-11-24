"""
ConduitCLI is our conduit library as a CLI application.

Customize the query_function to specialize for various prompts / workflows while retaining archival and other functionalities.

To customize:
1. Define your own query function matching the QueryFunctionProtocol signature.
2. Pass your custom function to the ConduitCLI class upon instantiation.

This allows you to tailor the behavior of ConduitCLI while leveraging its existing features.
"""

from argparse import ArgumentParser, Namespace
from conduit.cli.config_loader import ConfigLoader
from conduit.cli.handlers import HandlerMixin
from conduit.cli.query_function import CLIQueryFunctionProtocol, default_query_function
from conduit.cli.printer import Printer
from conduit.progress.verbosity import Verbosity
from conduit.model.models.modelstore import ModelStore
from xdg_base_dirs import xdg_data_home, xdg_config_home, xdg_cache_home
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_NAME = "conduit"
DEFAULT_DESCRIPTION = "Conduit: The LLM CLI"
DEFAULT_QUERY_FUNCTION = default_query_function
DEFAULT_VERBOSITY = Verbosity.COMPLETE
DEFAULT_CACHE_SETTING = True
DEFAULT_PERSISTENT_SETTING = True
DEFAULT_PREFERRED_MODEL = "claude"  # But can be overridden by config


class ConduitCLI(HandlerMixin):
    """
    Main class for the Conduit CLI application.
    Combines argument parsing, configuration loading, and command handling.
    Attributes:
    - name: Name of the CLI application.
    - description: Description of the CLI application.
    - verbosity: Verbosity level for LLM responses.
    - preferred_model: Default LLM model to use.
    - config: Configuration dictionary loaded from ConfigLoader.
    - attr_mapping: Maps command-line argument names to internal attribute names.
    - command_mapping: Maps command-line argument names to their respective handler method names.
    - flags: Dictionary to hold parsed flag values.
    - parser: ArgumentParser instance for parsing command-line arguments.
    - args: Parsed arguments from the command line.
    - stdin: Captured standard input if piped.
    - query_function: Function to handle queries, adhering to QueryFunctionProtocol.
    - cache: Boolean indicating whether to use caching for LLM responses.
    Methods:
    - setup_parser(): Sets up the argument parser based on the loaded configuration.
    Methods from HandlerMixin:
    - validate_handlers(): Validates that all handlers specified in the config are implemented.
    - all handler methods (e.g., handle_history, handle_wipe, etc.) should be implemented in this class.
    """

    def __init__(
        self,
        name: str = "conduit",
        description: str = DEFAULT_DESCRIPTION,
        query_function: CLIQueryFunctionProtocol = DEFAULT_QUERY_FUNCTION,
        verbosity: Verbosity = DEFAULT_VERBOSITY,
        cache: bool = DEFAULT_CACHE_SETTING,
        persistent: bool = DEFAULT_PERSISTENT_SETTING,
        system_message: str | None = None,  # If None, load from config or default to ""
        preferred_model: str | None = None,
    ):
        # Parameters
        self.name: str = name  # Name of the CLI application
        self.description: str = description  # description of the CLI application
        # Query function -- must adhere to QueryFunctionProtocol
        self.query_function: CLIQueryFunctionProtocol = (
            query_function  # function to handle queries
        )
        assert isinstance(query_function, CLIQueryFunctionProtocol), (
            "query_function must adhere to QueryFunctionProtocol"
        )
        # Configs
        self.verbosity: Verbosity = verbosity  # verbosity level for LLM responses
        self.cache: bool = cache  # whether to use caching for LLM responses
        # Get XDG paths
        self.history_file: Path
        self.config_dir: Path
        self.cache_file: Path
        self.history_file, self.config_dir, self.cache_file = (
            self._construct_xdg_paths()
        )
        # Preferred models
        self.preferred_model = preferred_model
        self.preferred_model = self._get_preferred_model(preferred_model)
        # System message: three options:
        if system_message is not None:
            self.system_message = system_message
        else:
            self.system_message = self._get_system_message()
        # Persistence
        if persistent:
            logger.info(f"Using persistent history at {self.history_file}")
            from conduit.message.messagestore import MessageStore
            from conduit.sync import Conduit

            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            message_store = MessageStore(history_file=self.history_file)
            Conduit.message_store = message_store
        else:
            logger.info("Using in-memory history (no persistence)")
        # Cache
        if cache:
            logger.info(f"Using cache at {self.cache_file}")
            from conduit.sync import Model
            from conduit.cache.cache import ConduitCache

            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            Model.conduit_cache = ConduitCache()
        else:
            logger.info("Caching disabled")
        # Set up config
        self.attr_mapping: dict = {}
        self.command_mapping: dict = {}
        self.parser: ArgumentParser | None = None
        self.args: Namespace | None = None
        self.flags: dict = {}  # This will hold all the flag values after parsing
        self.config: dict = ConfigLoader().config
        self.printer: Printer = Printer()  # IO policy manager
        self._validate_handlers()  # from HandlerMixin

    def run(self):
        """
        Run the CLI application.
        """
        self._preconfigure_logging()  # Set log level early
        # Logging begins
        logger.info("Running ConduitCLI")
        self.stdin: str = self._get_stdin()  # capture stdin if piped
        # Setup parser and parse args
        self.parser = self._setup_parser()
        # If no args, print help and exit
        if len(sys.argv) == 1 and not self.stdin:
            self.parser.print_help(sys.stderr)
            sys.exit(1)
        # Parse args
        self._parse_args()

    def _get_preferred_model(self, preferred_model: str | None) -> str:
        """
        Validate preferred_model if provided, otherwise load from config or use default.
        """
        model_store = ModelStore()
        if preferred_model:
            # Validate provided preferred_model
            if model_store._validate_model(preferred_model):
                return model_store._validate_model(preferred_model)
            else:
                logger.warning(
                    f"Preferred model '{preferred_model}' is not available. Falling back to config or default."
                )
        # Load from config if available
        try:
            preferred_model_file = xdg_config_home() / self.name / "preferred_model"
            preferred_model_from_config = preferred_model_file.read_text().strip()
            if model_store._validate_model(preferred_model_from_config):
                logger.info(
                    f"Loaded preferred model '{preferred_model_from_config}' from config file."
                )
                return model_store._validate_model(preferred_model_from_config)
            else:
                logger.warning(
                    f"Preferred model '{preferred_model_from_config}' from config file is not available. Falling back to default."
                )
        except Exception as e:
            logger.info(
                "No preferred model file found in .config, assuming default preferred model."
            )
        # Default preferred model
        logger.info(f"Using default preferred model '{DEFAULT_PREFERRED_MODEL}'")
        return DEFAULT_PREFERRED_MODEL

    def _get_system_message(self) -> str:
        """
        Get the system message from config if it exists, otherwise return "".
        """
        try:
            system_message_file = (
                xdg_config_home() / self.name / "system_message.jinja2"
            )
            system_message = system_message_file.read_text()
            logger.info(f"Loaded system message from {system_message_file}")
            return system_message
        except Exception as e:
            logger.info(
                "No system message file found in .config, assuming no system message."
            )
            return ""

    def _preconfigure_logging(self):
        """
        Quick arg scan to set log level before full parsing.
        """
        root = logging.getLogger()

        # If already configured (library usage), respect existing config
        if root.handlers:
            logger.debug("Logging already configured, using existing setup")
            return

        # Determine level from CLI args or use default
        level = logging.WARNING  # default

        if "--log" in sys.argv:
            idx = sys.argv.index("--log")
            if idx + 1 < len(sys.argv):
                log_flag = sys.argv[idx + 1].lower()
                levels = {"d": logging.DEBUG, "i": logging.INFO, "w": logging.WARNING}
                level = levels.get(log_flag, logging.WARNING)

        # Configure once with the correct level
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d (%(funcName)s) - %(message)s",
            stream=sys.stderr,
        )

        # Silence noisy libraries
        for lib in ["markdown_it", "urllib3", "httpx", "httpcore"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

    def _construct_xdg_paths(self) -> tuple[Path, Path, Path]:
        """
        Construct XDG-compliant paths for history, config, and cache files.
        The name (self.name) is used as the application directory for each.

        Returns a tuple of Paths: (history_file, config_file, cache_file)

        Note: config_file is not currently used but reserved for future use.
        """
        history_file = xdg_data_home() / self.name / "history.json"
        config_file = xdg_config_home() / self.name / "config.yaml"
        cache_file = xdg_cache_home() / self.name / "cache.sqlite"

        return history_file, config_file, cache_file

    def _get_stdin(self) -> str:
        """
        Get implicit context from clipboard or other sources.
        """
        context = sys.stdin.read() if not sys.stdin.isatty() else ""
        return context

    def _coerce_query_input(self, query_input: str | list) -> str:
        """
        Coerce query input to a string.
        If input is a list, join with spaces.
        """
        if isinstance(query_input, list):
            coerced_query_input = " ".join(query_input)
        else:
            coerced_query_input = query_input
        return coerced_query_input

    def _setup_parser(self):
        """
        Setup the argument parser based on the configuration.
        """
        logger.info("Setting up argument parser")
        parser = ArgumentParser()
        parser.description = self.description
        self.attr_mapping = {}
        self.command_mapping = {}

        # Handle positional args (i.e. query string if provided)
        logger.info("Adding positional arguments to parser")
        for pos_arg in self.config.get("positional_args", []):
            dest = pos_arg.pop("dest")
            self.attr_mapping[dest] = dest
            parser.add_argument(
                dest,
                **pos_arg,
            )

        # Handle flags
        logger.info("Adding flags to parser")
        for flag in self.config["flags"]:
            abbrev = flag.pop("abbrev", None)
            name = flag.pop("name")

            args = [abbrev, name] if abbrev else [name]
            arg_name = name.lstrip("-").replace("-", "_")
            self.attr_mapping[arg_name] = arg_name  # Same as dest
            parser.add_argument(*args, **flag)

        # Handle commands
        logger.info("Adding commands to parser")
        command_group = parser.add_mutually_exclusive_group()
        for command in self.config["commands"]:
            handler = command.pop("handler")
            abbrev = command.pop("abbrev", None)
            name = command.pop("name")

            args = [abbrev, name] if abbrev else [name]
            arg_name = name.lstrip("-").replace("-", "_")
            self.command_mapping[arg_name] = handler
            command_group.add_argument(*args, **command)

        return parser

    def _parse_args(self):
        """
        Parse arguments and execute commands or prepare for query processing.
        Note that three args are hardcoded specially:
        - raw: sets overall IO policy to raw (impacts all output)
        - log: sets logging level (handled earlier)
        - query_input: positional arg for the query string

        Everything else is dynamically mapped based on config (args.json + handlers.py).
        """
        logger.info("Parsing arguments")
        self.args = self.parser.parse_args()
        logger.debug(f"Parsed args: {self.args}")

        # Set IO policy
        if self.args.raw:
            self.printer.set_raw(True)
            logger.info(f"Printer policy set to raw.")

        # Create flags dictionary
        logger.info("Creating flags dictionary")
        self.flags = {}
        for arg_name, attr_name in self.attr_mapping.items():
            if hasattr(self.args, arg_name):
                self.flags[attr_name] = getattr(self.args, arg_name)

        # Coerce query input to string if necessary
        self.flags["query_input"] = self._coerce_query_input(self.flags["query_input"])

        # Check if any commands were specified and execute them
        logger.info("Checking for commands to execute")
        for arg_name, handler_name in self.command_mapping.items():
            if getattr(self.args, arg_name, False):
                handler = getattr(self, handler_name)
                if getattr(self.args, arg_name) not in [True, False]:
                    handler(getattr(self.args, arg_name))
                else:
                    handler()
                return

        # If no commands were executed and we have query input, process it
        logger.info("No commands executed; checking for query input")
        if self.args.query_input:
            self.query_handler()

    # Debugging methods
    def _print_all_attrs(self, pretty: bool = True):
        """
        Debugging method to print all attributes of the instance.
        """
        logger.info("Printing all attributes of the instance")
        if pretty:
            self.printer.print_pretty(vars(self))
            return
        else:
            attrs = vars(self)
            for attr, value in attrs.items():
                self.printer.print_raw(f"{attr}: {value}")

    def _get_handler_for_command(self, command_name: str):
        """
        Given a command-line argument name, return the corresponding handler method.
        """
        logger.info(f"Getting handler for command: {command_name}")
        handler_name = self.command_mapping.get(command_name)
        if handler_name and hasattr(self, handler_name):
            return getattr(self, handler_name)
        return None
