from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import click


class CommandCollection(Protocol):
    def _register_commands(self) -> None:
        """Register CLI commands."""
        ...

    def attach(self, group: click.Group) -> click.Group:
        """Attach commands to a Click group."""
        ...
