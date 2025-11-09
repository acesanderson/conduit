"""
All of our REPL commands for Chat are dynamically registered using decorators.
This module defines the Command dataclass and the command decorator which registers commands.
Commands can return either a string (to be printed) or a UICommand (to trigger UI actions).

Example usage:
```python
@command("greet", 1, "hello", "hi")
def greet_user(name: str):
    \"\"\"Greet the user by name.\"\"\"
    print(f"Hello, {name}!")
```

Notes:
- a command can have 0, 1, or multiple parameters.
- the docstring of the function is used as the command description.
- aliases allow multiple names to trigger the same command.
"""

from dataclasses import dataclass
from conduit.chat.ui.ui_command import UICommand
from collections.abc import Callable
from typing import Literal, Any
from rich.console import RenderableType


CommandResult = RenderableType | UICommand | tuple[UICommand, Any]


@dataclass
class Command:
    """
    Metadata for a registered command.

    Params:
        name: Official command name (used in /help and after /)
        func: The function to execute for this command
        aliases: Alternative names that trigger this command
        param_count: Number of parameters the command accepts:
            0 - no parameters
            1 - single parameter
            "multi" - multiple parameters (list of strings)
    """

    name: str
    func: Callable
    aliases: tuple[str, ...]
    param_count: Literal[0, 1, "multi"]

    @property
    def description(self) -> str:
        """Extract description from function docstring."""
        return self.func.__doc__ or "No description available."

    def validate(self) -> None:
        """Ensure command configuration is valid."""
        import inspect

        sig = inspect.signature(self.func)
        params = [p for p in sig.parameters.values() if p.name != "self"]

        if self.param_count == 0 and len(params) != 0:
            raise ValueError(
                f"Command '{self.name}' with param_count=0 must take no parameters"
            )
        elif self.param_count == 1 and len(params) != 1:
            raise ValueError(
                f"Command '{self.name}' with param_count=1 must take exactly one parameter"
            )
        elif self.param_count == "multi" and len(params) != 1:
            raise ValueError(
                f"Command '{self.name}' with param_count='multi' must take exactly one parameter (list[str])"
            )

    def execute(self, args: list[str] | None) -> CommandResult:
        """Execute command and return output string (if any)."""
        if self.param_count == 0:
            return self.func()
        elif self.param_count == 1:
            return self.func(args[0])
        else:
            return self.func(args)


# Our decorator
def command(
    name: str, param_count: Literal[0, 1, "multi"] = 0, aliases: list[str] | None = None
):
    """
    Register a command with explicit name, parameter count, and aliases.

    Args:
        name: Official command name (used in /help and after /)
        param_count: 0 (no params), 1 (single param), or "multi" (list of params)
        *aliases: Alternative names that trigger this command
    """

    def decorator(func: Callable) -> Callable:
        cmd = Command(
            name=name,
            func=func,
            aliases=tuple(aliases) if aliases else (),
            param_count=param_count,
        )
        cmd.validate()
        func._command = cmd
        return func

    return decorator
