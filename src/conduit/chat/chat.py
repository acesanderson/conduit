from conduit.progress.verbosity import Verbosity
from conduit.chat.registry import CommandRegistry
from conduit.chat.handlers import Handlers
from conduit.chat.app import ChatApp
from conduit.model.model import Model
from conduit.message.messagestore import MessageStore
from rich.console import Console


class ConduitChat(CommandRegistry, Handlers):
    """
    Command registry with handler methods mixed in.
    This combines the command infrastructure (CommandRegistry)
    with the actual command implementations (Handlers).
    """

    pass


def create_chat_app(
    preferred_model: str,
    welcome_message: str,
    system_message: str,
    message_store: MessageStore | None = None,
    console: Console | None = None,
    verbosity: Verbosity = Verbosity.PROGRESS,
) -> ChatApp:
    """
    Factory function to create a fully configured ChatApp.

    Args:
        preferred_model: Model name to use (e.g., "claude-sonnet-4")
        welcome_message: Message to display on startup
        system_message: System prompt for the conversation
        message_store: Optional message store (creates new if None)
        console: Optional Rich console (creates new if None)

    Returns:
        Configured ChatApp ready to run
    """
    # Create dependencies
    model = Model(preferred_model)
    message_store = message_store or MessageStore()
    console = console or Console()

    # Create registry with handlers
    registry = ConduitChat()

    # Inject dependencies into registry (handlers need these)
    registry.model = model
    registry.console = console
    registry.message_store = message_store
    registry.system_message = system_message
    registry.verbosity = verbosity

    # Create app with all dependencies
    app = ChatApp(
        registry=registry,
        model=model,
        console=console,
        message_store=message_store,
        welcome_message=welcome_message,
        system_message=system_message,
        verbosity=verbosity,
    )

    return app
