"""
Mixin class to manage keybindings for EnhancedInput class.
"""

from prompt_toolkit.key_binding import KeyBindings


class KeyBindingsRepo:
    """
    Mixin providing Escape-key keybindings for EnhancedInput terminal UI.

    Defines a repository of keyboard shortcuts using Escape as the leader key,
    enabling quick access to common chat operations (exit, clear history, toggle
    multiline mode, etc.) without interrupting natural text input. Requires mixed-in
    class to provide `console`, `engine`, `multiline_mode`, and `exit()` method.
    """

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        # <Esc>d → exit
        @kb.add("escape", "d")
        def exit_app(event):
            """
            Exit the application
            """
            self.exit()

        # <Esc>n → wipe message history (new chat)
        @kb.add("escape", "n")
        def new_chat(event):
            """
            Start a new chat (wipe history)
            """
            if self.engine:
                self.engine.conversation.wipe()
                self.console.print("[green]Message history cleared.[/green]")

        # <Esc>h → show keybindings help
        @kb.add("escape", "h")
        def show_keybindings(event):
            """
            Show available keybindings
            """
            from rich.table import Table

            table = Table(
                show_header=True, header_style="bold cyan", title="Keybindings"
            )
            table.add_column("Key", style="green")
            table.add_column("Action", style="yellow")

            table.add_row("<Esc>d", "Exit application")
            table.add_row("<Esc>n", "New chat (wipe history)")
            table.add_row("<Esc>h", "Show this help")
            table.add_row("<Esc>m", "Show model card")
            table.add_row("<Esc><Enter>", "Toggle multiline mode")

            self.console.print(table)

        # <Esc>m → show model card
        @kb.add("escape", "m")
        def show_model_card(event):
            """
            Display current model information
            """
            from conduit.core.model.models.modelstore import ModelStore

            ms = ModelStore()
            model_spec = ms.get_model(self.engine.params.model)
            model_spec.card

        # <Esc><Enter> → toggle multiline mode
        @kb.add("escape", "enter")
        def toggle_multiline(event):
            """
            Toggle multiline input mode
            """
            self.multiline_mode = not self.multiline_mode
            mode_status = "enabled" if self.multiline_mode else "disabled"
            self.console.print(f"[cyan]Multiline mode {mode_status}[/cyan]")

        return kb
