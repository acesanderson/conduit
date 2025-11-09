from conduit.chat.engine.dispatcher import CommandDispatcher
from conduit.chat.engine.handlers import CommandHandlers
from conduit.chat.engine.ui_handlers import UICommandHandlers


class ConduitEngine(CommandDispatcher, CommandHandlers, UICommandHandlers):
    """
    Unified command orchestrator combining parsing, routing, and execution for chat applications.

    Merges three independent mixins into a single interface: CommandDispatcher (parses and routes
    slash-prefixed commands), CommandHandlers (implements content commands like /help, /set model),
    and UICommandHandlers (implements UI commands like /clear, /exit). Manages LLM dependencies
    (model, message_store, system_message, verbosity) passed during initialization.

    Usage:
        engine = ConduitEngine(
            model="claude-3-haiku",
            message_store=MessageStore(),
            system_message="You are helpful.",
            verbosity=Verbosity.PROGRESS
        )
        output = engine.execute_command("/help")  # Returns help table
        response = engine.handle_query("What is AI?")  # Returns LLM response

    Attributes:
        model: Active language model instance for queries.
        message_store: Persistent message history and context manager.
        system_message: System prompt for LLM conversations.
        verbosity: Output verbosity level (SILENT through DEBUG).

    Raises:
        ValueError: If command name conflicts or invalid arguments provided.
        KeyboardInterrupt: If user cancels during model query.
    """

    pass
