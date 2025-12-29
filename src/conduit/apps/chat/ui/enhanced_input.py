"""
Our existing REPL application uses a basic input prompt. We want to enhance it with Prompt Toolkit features.
- [ ] 1.0 Spinner
- [x] 1.1 Command History with Persistence
- [x] 1.2 Tab Completion for Commands
- [x] 1.3 Multi-line Input
- [x] 1.4 Basic Key Bindings
- [x] 2.1 Bottom Toolbar
- [x] 2.2 Dynamic Toolbar
    - [x] refreshing data
    - [ ] app state (i.e. model, tokens, etc.)
- [ ] 3.1 Nested Completers
- [ ] 3.2 Fuzzy Command Matching
- [ ] 4.2 History View Keybinding
- [ ] 5.1 Clipboard Integration
- [ ] 5.2 Quick Save
- [ ] 6.1 Simple Expansion
- [ ] 8.1 Split Layout
"""

from __future__ import annotations
from conduit.apps.chat.ui.input_interface import InputInterface
from conduit.apps.chat.ui.ui_command import UICommand
from conduit.apps.chat.ui.keybindings import KeyBindingsRepo
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from typing import override, TYPE_CHECKING
from collections.abc import Iterable
import re
from pathlib import Path
import io
import shutil

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout

from conduit.core.model.models.modelstore import ModelStore

if TYPE_CHECKING:
    from conduit.apps.chat.engine.async_engine import ChatEngine

# Precompile regex pattern (copied from BasicInput)
style_pattern = re.compile(r"\[/?[a-zA-Z0-9_ ]+\]")

# 1.1: Define a path for the persistent history file
HISTORY_FILE = Path.home() / ".conduit_chat_history"


class CommandCompleter(Completer):
    """
    A prompt_toolkit completer that suggests commands from the ChatEngine.
    """

    def __init__(self, engine: ChatEngine):
        self.engine: ChatEngine = engine
        self._all_commands: dict[str, str] | None = None

    def _get_commands(self):
        """Cache command and alias names."""
        if self._all_commands is None:
            self._all_commands = {}
            registered_commands = self.engine.commands
            for name, handler in registered_commands.items():
                # name is like "/help" or "/set model"
                command_name = name[1:]  # remove leading '/'
                description = ""
                if handler.__doc__:
                    # Get the first line of the docstring
                    description = handler.__doc__.strip().split("\n")[0]
                self._all_commands[command_name] = description

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        """
        Yield completion suggestions.
        """
        self._get_commands()
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        # If user has typed a full command and a space, do not complete.
        if text[1:] in self._all_commands and text.endswith(" "):
            return

        command_text = text[1:]

        # Suggest primary commands
        for name, description in self._all_commands.items():
            if name.startswith(command_text):
                yield Completion(
                    text=f"/{name}",
                    start_position=-len(text),  # Replace the whole thing user typed
                    display=f"/{name}",
                    display_meta=description,
                )


class EnhancedInput(InputInterface, KeyBindingsRepo):
    """
    Enhanced input using Prompt Toolkit for features like persistent history.

    Output strategy ("boring and stable"):
    - While the prompt is active, prompt_toolkit owns terminal output.
    - Rich renderables are rendered to ANSI strings in-memory (no stdout side effects).
    - ANSI strings are printed via prompt_toolkit's safe printing mechanism (run_in_terminal).
    """

    def __init__(self, console: Console):
        self.console: Console = console
        self._model_spec_cache = {}
        self._model_store = ModelStore()

        # We need extra context about commands for tab completion, as well as the model name.
        self.engine: ChatEngine | None = None

        # Track multiline mode state
        self.multiline_mode = False

        # Create key bindings
        self.kb = self._create_key_bindings()

        self.session: PromptSession = PromptSession(
            # Create a session that uses a persistent file history
            history=FileHistory(str(HISTORY_FILE)),
            vi_mode=True,
            # Tab Completion Setup
            completer=None,
            complete_while_typing=True,  # Show suggestions as you type
            # Multi-line Input Setup
            multiline=Condition(lambda: self.multiline_mode),
            prompt_continuation=".. ",  # Optional: Prompt for 2nd+ lines
            # Toolbar
            bottom_toolbar=self.get_toolbar_text,
            # Key bindings
            key_bindings=self.kb,
        )

        # Basic styling for the prompt (approximates BasicInput's gold3)
        self.style = Style.from_dict(
            {
                "prompt": "bold #ffaf00",
                # --- completion menu ---
                "completion-menu.completion": "bg:#005f5f #ffffff",
                "completion-menu.completion.current": "bg:#008787 #ffffff",
                "scrollbar.background": "bg:#005f5f",
                "scrollbar.button": "bg:#008787",
            }
        )

    def _get_model_spec(self, model_name: str):
        if model_name not in self._model_spec_cache:
            self._model_spec_cache[model_name] = self._model_store.get_model(model_name)
        return self._model_spec_cache[model_name]

    def get_toolbar_text(self):
        if not self.engine:
            return [("class:bottom-toolbar", " <Esc>h for keybindings")]

        model_name = self.engine.params.model or "unknown"
        context_length = len(self.engine.conversation.messages)

        try:
            model_spec = self._get_model_spec(model_name)
            context_window = model_spec.context_window if model_spec else 0
        except Exception:
            context_window = 0

        multiline_indicator = " [MULTILINE]" if self.multiline_mode else ""

        return [
            (
                "class:bottom-toolbar",
                f" <Esc>h for keybindings | {model_name} | Context: {context_length}/{context_window}{multiline_indicator} ",
            )
        ]

    def set_engine(self, engine: ChatEngine) -> None:
        """
        Set the engine to enable tab completion.
        This is called by the factory after the engine is created.
        """
        self.engine = engine
        self.session.completer = CommandCompleter(engine)

    async def get_input(self, prompt: str = ">> ") -> str:
        """
        Get user input using Prompt Toolkit.
        Use patch_stdout to prevent stray prints/logging from corrupting the prompt.
        """
        styled_prompt_message = [("class:prompt", prompt)]
        with patch_stdout():
            return await self.session.prompt_async(
                styled_prompt_message, style=self.style
            )

    def _get_render_width(self) -> int:
        """
        Determine a stable width for Rich rendering.

        Prefer prompt_toolkit's notion of terminal width when the app exists,
        otherwise fall back to the current terminal size / rich console width.
        """
        try:
            if self.session.app:
                size = self.session.app.output.get_size()
                if size and getattr(size, "columns", None):
                    return max(20, int(size.columns))
        except Exception:
            pass

        try:
            cols = shutil.get_terminal_size().columns
            return max(20, int(cols))
        except Exception:
            return max(20, int(getattr(self.console, "width", 80)))

    def _render_to_ansi(self, renderable: RenderableType) -> str:
        """
        Render a Rich renderable to an ANSI-escaped string in memory.
        No stdout side effects.
        """
        buf = io.StringIO()
        width = self._get_render_width()

        render_console = Console(
            file=buf,
            force_terminal=True,
            color_system="truecolor",
            width=width,
            markup=True,
            emoji=True,
            highlight=False,
        )
        render_console.print(renderable)
        return buf.getvalue()

    def _print_ansi_safely(self, ansi_text: str) -> None:
        """
        Print ANSI text above the current prompt without corrupting the UI.
        """

        def _do_print() -> None:
            # ANSI() parses escape sequences into prompt_toolkit formatted text.
            print_formatted_text(ANSI(ansi_text), end="")

        run_in_terminal(_do_print)

    @override
    def show_message(self, message: RenderableType, style: str = "info") -> None:
        """
        Display a message without breaking the prompt.

        Behavior:
        - If prompt_toolkit app is running: render (Rich) -> ANSI string -> print safely.
        - Otherwise: fallback to direct Rich printing for startup / non-interactive output.
        """
        if self.session.app:
            # Normalize strings into Rich renderables so we keep styling consistent.
            if isinstance(message, str):
                if style_pattern.search(message):
                    renderable: RenderableType = message  # rich markup string
                else:
                    renderable = Markdown(message)
            else:
                renderable = message

            ansi_text = self._render_to_ansi(renderable)
            self._print_ansi_safely(ansi_text)
            return

        # Fallback for initial messages before prompt_toolkit app is fully running.
        if isinstance(message, str):
            if style_pattern.search(message):
                self.console.print(message)
            else:
                self.console.print(Markdown(message))
        else:
            self.console.print(message)

    # UI commands
    @override
    def execute_ui_command(self, command: UICommand) -> None:
        if command == UICommand.CLEAR_SCREEN:
            self.clear_screen()
        elif command == UICommand.CLEAR_HISTORY_FILE:
            self.clear_history_file()
        elif command == UICommand.EXIT:
            self.exit()
        else:
            raise NotImplementedError(
                f"UI command {command} not implemented in EnhancedInput."
            )

    @override
    def clear_screen(self) -> None:
        """
        Clear the screen.
        """
        self.console.clear()

    @override
    def clear_history_file(self) -> None:
        """
        Clear the persistent history file
        """
        try:
            HISTORY_FILE.unlink()
            self.show_message(f"[green]Cleared history file at {HISTORY_FILE}.[/green]")
        except FileNotFoundError:
            self.show_message(
                f"[yellow]No history file found at {HISTORY_FILE}.[/yellow]"
            )

        # Recreate empty history file
        HISTORY_FILE.touch()

    @override
    def exit(self) -> None:
        """
        Exit the application
        """
        import sys

        sys.exit(0)
