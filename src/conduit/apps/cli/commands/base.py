import click
from conduit.apps.cli.commands.commands import CommandCollection
from typing import override


class BaseCommands(CommandCollection):
    """
    Attachable command collection for Conduit CLI.
    Inject onto any Click group via `attach(group)`.
    """

    def __init__(self):
        self._commands: list[click.Command] = []
        self._register_commands()

    @override
    def _register_commands(self):
        """Define all commands as bound methods."""

        @click.command()
        @click.option("-m", "--model", type=str, help="Specify the model to use.")
        @click.option("-L", "--local", is_flag=True, help="Use local HeadwaterServer.")
        @click.option("-r", "--raw", is_flag=True, help="Print raw output.")
        @click.option("-t", "--temperature", type=float, help="Temperature (0.0-1.0).")
        @click.option(
            "-c", "--chat", is_flag=True, help="Enable chat mode with history."
        )
        @click.option("-a", "--append", type=str, help="Append to query after stdin.")
        @click.option(
            "-p", "--prepend", type=str, help="Prepend to query before stdin."
        )
        @click.option(
            "-f", "--file", type=click.Path(exists=True), help="Read input from file."
        )
        @click.argument("query_input", nargs=-1)
        @click.pass_context
        def query(
            ctx: click.Context,
            model: str,
            local: bool,
            raw: bool,
            temperature: float,
            chat: bool,
            append: str,
            prepend: str,
            file: str,
            query_input: str,
        ):
            """Execute a query (default command)."""
            raise NotImplementedError

        @click.command()
        def history():
            """View message history."""
            raise NotImplementedError

        @click.command()
        def wipe():
            """Wipe message history."""
            raise NotImplementedError

        @click.command()
        def ping():
            """Ping the Headwater server."""
            raise NotImplementedError

        @click.command()
        def status():
            """Get Headwater server status."""
            raise NotImplementedError

        @click.command()
        def shell():
            """Enter interactive shell mode."""
            raise NotImplementedError

        @click.command()
        def last():
            """Get the last message."""
            raise NotImplementedError

        @click.command()
        @click.argument("index", type=int)
        def get(index):
            """Get a specific message from history."""
            raise NotImplementedError

        @click.command()
        def config():
            """View current configuration."""
            raise NotImplementedError

        self._commands = [
            query,
            history,
            wipe,
            ping,
            status,
            shell,
            last,
            get,
            config,
        ]

    @override
    def attach(self, group: click.Group) -> click.Group:
        """Attach all commands to a Click group."""
        for cmd in self._commands:
            group.add_command(cmd)
        return group
