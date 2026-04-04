from conduit.apps.cli.commands.base_commands import BaseCommands
from conduit.apps.cli.commands.batch_commands import batch_command
from conduit.apps.cli.commands.cache_commands import cache
from conduit.apps.cli.commands.models_commands import models_command
from conduit.apps.cli.commands.update_commands import update
from conduit.apps.cli.cli_class import ConduitCLI
import sys


def query_entrypoint():
    """
    Shortcut entry point for 'ask'.
    Takes us directly to "conduit query --persist ..."
    Persistence is on by default for interactive 'ask' usage.
    """

    # Safety check: don't inject if the user (weirdly) typed 'ask query'
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        pass
    else:
        sys.argv.insert(1, "query")

    # Inject --persist so interactive 'ask' calls persist to the message store
    if "--persist" not in sys.argv:
        sys.argv.insert(2, "--persist")

    # Now run the main app exactly as normal
    main()


def models_entrypoint():
    """
    Shortcut entry point for 'models'.
    Takes us directly to "conduit models ..."
    """
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        pass
    else:
        sys.argv.insert(1, "models")

    main()


def main():
    conduit_cli = ConduitCLI()
    commands = BaseCommands()
    conduit_cli.attach(commands)
    conduit_cli.cli.add_command(cache)
    conduit_cli.cli.add_command(models_command)
    conduit_cli.cli.add_command(batch_command)
    conduit_cli.cli.add_command(update)
    conduit_cli.run()


if __name__ == "__main__":
    main()
