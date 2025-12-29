"""
Mixin class to manage keybindings for EnhancedInput class.
"""

from prompt_toolkit.key_binding import KeyBindings


class KeyBindingsRepo:
    """
    Mixin providing Escape-key keybindings for EnhancedInput terminal UI.

    IMPORTANT OUTPUT RULE:
    In EnhancedInput mode, do not call self.console.print() directly from keybindings.
    Route all output through self.show_message() so prompt_toolkit can print safely.
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
                self.show_message("[green]Message history cleared.[/green]")

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

            self.show_message(table)

        # <Esc>m → show model card
        @kb.add("escape", "m")
        def show_model_card(event):
            """
            Display current model information
            """
            if not self.engine:
                self.show_message("[yellow]Engine not ready yet.[/yellow]")
                return

            from conduit.core.model.models.modelstore import ModelStore

            ms = ModelStore()
            model_spec = ms.get_model(self.engine.params.model)

            # Best effort: show something useful.
            # If model_spec.card is a rich renderable or string, show_message can handle it.
            try:
                card = getattr(model_spec, "card", None)
                if card:
                    self.show_message(card)
                else:
                    self.show_message(f"[cyan]Model:[/cyan] {self.engine.params.model}")
            except Exception as e:
                self.show_message(f"[red]Error showing model card: {e}[/red]")

        # <Esc><Enter> → toggle multiline mode
        @kb.add("escape", "enter")
        def toggle_multiline(event):
            """
            Toggle multiline input mode
            """
            self.multiline_mode = not self.multiline_mode
            mode_status = "enabled" if self.multiline_mode else "disabled"
            self.show_message(f"[cyan]Multiline mode {mode_status}[/cyan]")

        return kb
