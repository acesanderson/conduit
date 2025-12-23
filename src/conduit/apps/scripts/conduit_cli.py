from conduit.apps.cli.commands.base_commands import BaseCommands
from conduit.apps.cli.cli_class import ConduitCLI
import sys


def query_entrypoint():
    """
    Shortcut entry point for 'ask'.
    Takes us directly to "conduit query ..."
    """

    # Safety check: don't inject if the user (weirdly) typed 'ask query'
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        pass
    else:
        sys.argv.insert(1, "query")

    # Now run the main app exactly as normal
    main()


def main():
    conduit_cli = ConduitCLI()
    commands = BaseCommands()
    conduit_cli.attach(commands)
    conduit_cli.run()


if __name__ == "__main__":
    main()
