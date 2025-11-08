from conduit.chat.command_dispatcher import CommandDispatcher
from conduit.chat.command_handlers import CommandHandlers


class ConduitEngine(CommandDispatcher, CommandHandlers):
    """
    Command dispatcher with handler methods mixed in.
    This combines the command infrastructure (CommandRegistry)
    with the actual command implementations (Handlers).
    Also all of our LLM dependencies are passed here.
    """

    pass
