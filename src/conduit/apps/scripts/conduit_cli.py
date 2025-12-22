from conduit.apps.cli.commands.base import BaseCommands
from conduit.apps.cli.cli_class import ConduitCLI


def main():
    conduit_cli = ConduitCLI()
    commands = BaseCommands()
    conduit_cli.attach(commands)
    conduit_cli.run()


if __name__ == "__main__":
    main()
