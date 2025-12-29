import asyncio
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style

# Assumes the backend code is saved as backend.py
from backend import MockController, AppMode, Role

# --- 1. Renderer ---


def get_formatted_text(controller):
    """
    Iterates over controller.conversation.events to render the transcript.
    Highlights the selected event based on state.
    """

    def _get_text():
        result = []
        # Fallback if conversation is empty or selection is None
        if not controller.conversation.events:
            return [("", "No events.")]

        selected_id = controller.state.selected_event_id

        for event in controller.conversation.events:
            is_selected = event.id == selected_id

            # Base style
            style = ""
            if is_selected:
                style = "class:selected"
            elif event.role == Role.ASSISTANT:
                style = "class:assistant"
            elif event.role == Role.USER:
                style = "class:user"

            # Branch indicators
            prefix = "  "
            if event.total_branches > 1:
                prefix = f"[{event.branch_id + 1}/{event.total_branches}] "

            # Role marker
            role_str = "YOU" if event.role == Role.USER else "TAP"

            # Compose line
            # We use distinct tokens for role vs content to allow fine-grained styling
            result.append((style + " bold", f"{prefix}{role_str}: "))
            result.append((style, f"{event.content}\n\n"))

        return result

    return _get_text


# --- 2. Key Bindings ---


def create_keybindings(controller, input_buffer):
    kb = KeyBindings()

    @Condition
    def is_normal_mode():
        return controller.state.mode == AppMode.NORMAL

    @Condition
    def is_insert_mode():
        return controller.state.mode == AppMode.INSERT

    # -- Navigation (Normal Mode) --
    @kb.add("j", filter=is_normal_mode)
    def _(event):
        controller.move_selection(1)

    @kb.add("k", filter=is_normal_mode)
    def _(event):
        controller.move_selection(-1)

    # -- Mode Switching --
    @kb.add("o", filter=is_normal_mode)
    def _(event):
        # Open main input (Spec 4.1)
        controller.set_mode(AppMode.INSERT)
        event.app.layout.focus(input_buffer)

    @kb.add("escape", filter=is_insert_mode)
    def _(event):
        # Return to normal (Spec 2.4)
        controller.set_mode(AppMode.NORMAL)
        # Focus the transcript (assumed first child)
        event.app.layout.focus(event.app.layout.children[0].children[0])

    # -- Input Submission --
    # NOTE: 'c-enter' is not a valid PTK key. We bind Alt+Enter and Ctrl+J (EOF/Submit).
    @kb.add("escape", "enter", filter=is_insert_mode)
    @kb.add("c-j", filter=is_insert_mode)
    def _(event):
        text = input_buffer.document.text
        if not text.strip():
            return

        # Wrap in asyncio.create_task as requested
        asyncio.create_task(controller.submit_input(text))

        # UI cleanup
        input_buffer.reset()
        controller.set_mode(AppMode.NORMAL)
        # Refocus transcript
        event.app.layout.focus(event.app.layout.children[0].children[0])

    @kb.add("c-c")
    def _(event):
        event.app.exit()

    return kb


# --- 3. Main Application Assembly ---


async def main():
    # Instantiate Controller
    controller = MockController()

    # Input Buffer
    input_buffer = Buffer(multiline=True)

    # UI Components
    transcript_control = FormattedTextControl(
        text=get_formatted_text(controller), focusable=True
    )

    # Transcript Window
    transcript_window = Window(
        content=transcript_control,
        wrap_lines=True,
        cursorline=False,  # We handle highlighting manually in renderer
        always_hide_cursor=True,
    )

    input_window = Window(
        content=BufferControl(buffer=input_buffer), height=4, style="class:input-area"
    )

    # Separation Line
    status_bar = Window(height=1, char="-", style="class:line")

    # Layout Hierarchy
    root_container = FloatContainer(
        content=HSplit([transcript_window, status_bar, input_window]), floats=[]
    )

    layout = Layout(root_container, focused_element=transcript_window)

    # Styles
    style = Style.from_dict(
        {
            "selected": "reverse",
            "assistant": "#00ff00",
            "user": "#ffffff",
            "line": "#444444",
            "input-area": "#cccccc",
        }
    )

    # Build App
    app = Application(
        layout=layout,
        key_bindings=create_keybindings(controller, input_buffer),
        style=style,
        full_screen=True,
        mouse_support=False,
    )

    # Register redraw hook
    controller.register_redraw_callback(app.invalidate)

    await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
