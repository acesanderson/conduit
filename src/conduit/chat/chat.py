"""
Chat interface factory and command registry combining user-facing conversation management with extensible command handling infrastructure. This module bridges three core components—CommandRegistry (command parsing and execution), Handlers (command implementations like `/help`, `/exit`), and ChatApp (REPL event loop)—into a unified ConduitChat class that powers interactive multi-turn conversations.

The primary factory function `create_chat_app` instantiates a fully configured ChatApp by assembling dependencies (Model, MessageStore, InputInterface) and injecting them into a ConduitChat instance that serves as both command registry and handler provider. This design enables users to launch a complete chat session with a single function call while allowing command discovery and routing to happen automatically through mixin inheritance.
"""

from conduit.progress.verbosity import Verbosity
from conduit.chat.registry import CommandRegistry
from conduit.chat.handlers import Handlers
from conduit.chat.app import ChatApp
from conduit.chat.ui.input_interface import InputInterface
from conduit.model.model import Model
from conduit.message.messagestore import MessageStore


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
    input_interface: InputInterface,
    message_store: MessageStore | None = None,
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

    # Create registry with handlers
    registry = ConduitChat()

    # Inject dependencies into registry (handlers need these)
    registry.model = model
    registry.input_interface = input_interface
    registry.message_store = message_store
    registry.system_message = system_message
    registry.verbosity = verbosity

    # Create app with all dependencies
    app = ChatApp(
        registry=registry,
        model=model,
        input_interface=input_interface,
        message_store=message_store,
        welcome_message=welcome_message,
        system_message=system_message,
        verbosity=verbosity,
    )

    return app
