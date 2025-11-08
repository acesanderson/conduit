"""
Chat interface factory and command registry combining user-facing conversation management with extensible command handling infrastructure. This module bridges three core components—CommandRegistry (command parsing and execution), Handlers (command implementations like `/help`, `/exit`), and ChatApp (REPL event loop)—into a unified ConduitChat class that powers interactive multi-turn conversations.

The primary factory function `create_chat_app` instantiates a fully configured ChatApp by assembling dependencies (Model, MessageStore, InputInterface) and injecting them into a ConduitChat instance that serves as both command registry and handler provider. This design enables users to launch a complete chat session with a single function call while allowing command discovery and routing to happen automatically through mixin inheritance.
"""

from conduit.progress.verbosity import Verbosity
from conduit.chat.app import ChatApp
from conduit.chat.engine import ConduitEngine
from conduit.chat.ui.input_interface import InputInterface
from conduit.message.messagestore import MessageStore


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
    ## Message_store
    message_store = message_store or MessageStore()
    ## Create dispatch with handlers
    engine = ConduitEngine(
        model=preferred_model,
        message_store=message_store,
        system_message=system_message,
        verbosity=verbosity,
    )

    # Prompt toolkit based input interfaces need the registry
    # Create app with all dependencies
    app = ChatApp(
        engine=engine,
        input_interface=input_interface,
        welcome_message=welcome_message,
    )

    return app
