from conduit.apps.cli.commands.base import BaseCommands
from conduit.apps.cli.cli_class import ConduitCLI


if __name__ == "__main__":
    conduit_cli = ConduitCLI()
    commands = BaseCommands()
    conduit_cli.attach(commands)
    conduit_cli.run()
