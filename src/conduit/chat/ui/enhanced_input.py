"""
Our existing REPL application uses a basic input prompt. We want to enhance it with Prompt Toolkit features.

- [ ] 1.0 Spinner
- [x] 1.1 Command History with Persistence
- [x] 1.2 Tab Completion for Commands
- [x] 1.3 Multi-line Input
- [ ] 1.4 Basic Key Bindings
- [x] 2.1 Bottom Toolbar
- [ ] 2.2 Dynamic Toolbar
    - [ ] app state (i.e. model, tokens, etc.)
    - [ ] refreshing data
- [ ] 3.1 Nested Completers
- [ ] 3.2 Fuzzy Command Matching
- [ ] 4.2 History View Keybinding
- [ ] 5.1 Clipboard Integration
- [ ] 5.2 Quick Save
- [ ] 6.1 Simple Expansion
- [ ] 8.1 Split Layout
"""

from conduit.chat.ui.input_interface import InputInterface
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from typing import override, TYPE_CHECKING
from collections.abc import Iterable
import re
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from conduit.chat.registry import CommandRegistry

# Precompile regex pattern (copied from BasicInput)
style_pattern = re.compile(r"\[/?[a-zA-Z0-9_ ]+\]")

# 1.1: Define a path for the persistent history file
HISTORY_FILE = Path.home() / ".conduit_chat_history"


class CommandCompleter(Completer):
    """
    A prompt_toolkit completer that suggests commands from a CommandRegistry.
    """

    def __init__(self, registry: "CommandRegistry"):
        self.registry = registry
        self._all_commands = None
        self._all_aliases = None

    def _get_commands(self):
        """Cache command and alias names."""
        if self._all_commands is None:
            self._all_commands = {}
            self._all_aliases = {}
            commands = self.registry.get_all_commands()
            for cmd in commands:
                self._all_commands[cmd.name] = cmd.description.strip().split("\n")[0]
                for alias in cmd.aliases:
                    self._all_aliases[alias] = f"Alias for /{cmd.name}"

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """
        Yield completion suggestions.
        """
        self._get_commands()
        text = document.text_before_cursor

        # Only complete if it starts with / and is the first word
        if not text.startswith("/") or " " in text:
            return

        command_text = text[1:]  # The part after the '/'

        # Suggest primary commands
        for name, description in self._all_commands.items():
            if name.startswith(command_text):
                yield Completion(
                    text=f"/{name}",
                    start_position=-len(command_text),
                    display=f"/{name}",
                    display_meta=description,
                )

        # Suggest aliases
        for alias, description in self._all_aliases.items():
            if alias.startswith(command_text):
                yield Completion(
                    text=f"/{alias}",
                    start_position=-len(command_text),
                    display=f"/{alias}",
                    display_meta=description,
                )


class EnhancedInput(InputInterface):
    """
    Enhanced input using Prompt Toolkit for features like persistent history.
    """

    def __init__(self, console: Console):
        self.console: Console = console
        self.registry: "CommandRegistry" | None = None

        def get_toolbar_text():
            return [("class:bottom-toolbar", " Type /help for instructions. ")]

        # 1.1: Create a session that uses a persistent file history
        self.session: PromptSession = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            vi_mode=True,
            # Tab Completion Setup
            completer=None,
            complete_while_typing=True,  # Show suggestions as you type
            # Multi-line Input Setup
            multiline=True,  # <-- 1.3: Enable multi-line input
            prompt_continuation=".. ",  # Optional: Prompt for 2nd+ lines
            # Toolbar
            bottom_toolbar=get_toolbar_text,
        )

        # Basic styling for the prompt (approximates BasicInput's gold3)
        self.style = Style.from_dict(
            {
                "prompt": "bold #ffaf00",
                # --- NEW: Style for completion menu ---
                "completion-menu.completion": "bg:#005f5f #ffffff",
                "completion-menu.completion.current": "bg:#008787 #ffffff",
                "scrollbar.background": "bg:#005f5f",
                "scrollbar.button": "bg:#008787",
                # --- End New ---
            }
        )

    def set_registry(self, registry: "CommandRegistry"):
        """
        Set the command registry to enable tab completion.
        This is called by the factory after the registry is created.
        """
        self.registry = registry
        self.session.completer = CommandCompleter(registry)

    @override
    def get_input(self, prompt: str = ">> ") -> str:
        """
        Get user input using Prompt Toolkit.
        """
        styled_prompt_message = [("class:prompt", prompt)]

        return self.session.prompt(styled_prompt_message, style=self.style)

    @override
    def show_message(self, message: RenderableType, style: str = "") -> None:
        """
        Display a message (Copied from BasicInput)
        """
        if isinstance(message, str):
            if style:
                self.console.print(message, style=style)
                return
            if style_pattern.search(message):
                self.console.print(message)
                return
            else:
                message = Markdown(message)
                self.console.print(message)
        else:
            self.console.print(message)

    @override
    def clear_screen(self) -> None:
        """
        Clear the screen (Copied from BasicInput)
        """
        self.console.clear()
