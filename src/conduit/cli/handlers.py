"""
Some guidelines for this Mixin:
- all methods must have the signature (self, *args, **kwargs)
- if the config file specifies a type for a command or flag, it must be one of: str, int, float, bool, and the method must handle that type accordingly
- the method names must match the handler names in the config file exactly, a la "handle_history", "handle_wipe", etc.
"""

from conduit.sync import Conduit, Verbosity
from conduit.cli.printer import Printer
from conduit.cli.query_function import CLIQueryFunctionInputs
import logging
import sys

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VERBOSITY = Verbosity.PROGRESS


class HandlerMixin:
    def _validate_handlers(self):
        logger.debug("Validating handlers...")
        self.config: dict  # ConduitCLI has self.config (generated from ConfigLoader)
        for command in self.config["commands"]:
            handler_name = command.get("handler")
            if not hasattr(self, handler_name):
                raise ValueError(
                    f"Handler {handler_name} not implemented in {self.__class__.__name__}"
                )
            if not callable(getattr(self, handler_name)):
                raise ValueError(
                    f"Handler {handler_name} is not callable in {self.__class__.__name__}"
                )

    # Convenience methods
    def print_markdown(self, markdown_string: str):
        """
        Prints formatted markdown to the console.
        """
        self.printer.print_markdown(markdown_string)

    def grab_image_from_clipboard(self) -> tuple | None:
        """
        Attempt to grab image from clipboard; return tuple of mime_type and base64.
        """
        logger.info("Attempting to grab image from clipboard...")
        import os

        if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
            self.printer.print_pretty(
                "Image paste not available over SSH.", style="red"
            )
            return

        import warnings
        from PIL import ImageGrab
        import base64, io

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress PIL warnings
            image = ImageGrab.grabclipboard()

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")  # type: ignore
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            # Save for next query
            self.printer.print_pretty("Image captured!", style="green")
            # Build our ImageMessage
            image_content = img_base64
            mime_type = "image/png"
            return mime_type, image_content
        else:
            self.printer.print_pretty("No image detected.", style="red")

            sys.exit()

    def create_image_message(
        self, combined_query: str, mime_type: str, image_content: str
    ) -> "ImageMessage | None":
        logger.info("Creating ImageMessage...")
        if not image_content or not mime_type:
            return
        role = "user"
        text_content = combined_query

        from conduit.message.imagemessage import ImageMessage

        imagemessage = ImageMessage(
            role=role,
            text_content=text_content,
            image_content=image_content,
            mime_type=mime_type,
        )
        return imagemessage

    # Handlers -- should match config file exactly (per validate_handlers)
    def handle_history(self):
        """
        View message history and exit.
        """
        logger.info("Viewing message history...")
        Conduit.message_store.view_history()
        sys.exit()

    def handle_wipe(self):
        """
        Clear the message history after user confirmation.
        """
        logger.info("Wiping message history...")
        from rich.prompt import Confirm

        confirm = Confirm.ask(
            "[red]Are you sure you want to wipe the message history? This action cannot be undone.[/red]",
            default=False,
        )
        if confirm:
            Conduit.message_store.clear()
            self.printer.print_pretty("[green]Message history wiped.[/green]")
        else:
            self.printer.print_pretty("[yellow]Wipe cancelled.[/yellow]")

    def handle_shell(self):
        pass

    def handle_ping(self):
        from headwater_client.client.headwater_client import HeadwaterClient

        hc = HeadwaterClient()
        response = hc.ping()
        if response == True:
            response = "Pong!"
            self.printer.print_pretty(f"[green]{response}[/green]")
        else:
            response = "No response."
            self.printer.print_pretty(f"[red]{response}[/red]")

    def handle_status(self):
        from headwater_client.client.headwater_client import HeadwaterClient

        hc = HeadwaterClient()
        status = hc.get_status()
        self.printer.print_pretty(status)

    def handle_last(self):
        """
        Print the last message in the message store and exit.
        """
        logger.info("Viewing last message...")
        import sys

        # Get last message
        last_message = Conduit.message_store.last()
        # If no messages, inform user
        if not last_message:
            self.printer.print_pretty("[red]No messages in history.[/red]")
            sys.exit()
        # Print last message
        self.printer.print_markdown(str(last_message))
        sys.exit()

    def handle_get(self, index: int):
        pass

    def handle_config(self):
        """
        Print the current configuration and exit.
        """
        preferred_model = getattr(self, "preferred_model", "")
        system_message = getattr(self, "system_message", "")[:50]
        logger.info("Viewing configuration...")
        config_md = f"""
# Current Configuration
| Setting | Value |
|---------|-------|
| Preferred Model | {preferred_model} |
| System Message | {system_message}... |
| Message History | {"Enabled" if self.flags.get("chat") else "Disabled"} |
| Verbosity | {self.verbosity.name} |
"""
        self.printer.print_markdown(config_md)

    def query_handler(self):
        """
        Handle a query by combining query input, context, and append, then sending to model.

        Simplest usage:
            `conduit "What is the capital of France?"`
        Maximal usage:
            `cat "some_document.md" | conduit -q "Look at this doc." -a " Please summarize."`
        """
        logger.info("Handling query...")
        # Type hints since mixins confuse IDEs
        self.printer: Printer
        self.flags: dict[str, str | bool]
        self.stdin: str
        self.verbosity: Verbosity
        self.preferred_model: str

        # Grab our flags
        logger.debug("Grabbing flags...")
        ## NOTE: we need to implement temperature, image, and other flags here.
        chat = self.flags["chat"]
        user_defined_model = self.flags["model"]
        model_to_use = user_defined_model or self.preferred_model
        # Assemble the parts of the query
        logger.debug("Assembling query parts...")
        inputs = CLIQueryFunctionInputs(
            name=self.name,
            query_input=self.flags.get("query_input", ""),
            context=f"<context>{self.stdin}</context>" if self.stdin else "",
            append=self.flags.get("append") or "",
            system_message=getattr(self, "system_message", None),
            local=self.flags.get("local", False),
            preferred_model=model_to_use,
            verbose=self.verbosity,
            cache=self.cache,
        )
        # Start our spinner
        with self.printer.status(
            "[bold green]Thinking...[/bold green]", spinner="dots"
        ):
            # Our switch logic
            match chat:
                case True:  # Chat (with history), pretty print
                    inputs.include_history = True
                    logger.debug("Chat (with history), pretty print...")
                    response = self.query_function(inputs)
                    self.print_markdown(str(response.content))
                    sys.exit()
                case False:  # One-off request, pretty print
                    inputs.include_history = False
                    logger.debug("One-off request, pretty print...")
                    response = self.query_function(
                        inputs,
                    )
                    self.print_markdown(str(response.content))
                    sys.exit(None)
