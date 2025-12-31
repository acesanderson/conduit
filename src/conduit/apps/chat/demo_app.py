"""
Factory module for the Conduit chat application.
Now supports 'demo' mode for the split-screen TUI.
"""

from conduit.apps.chat.app import ChatApp
from conduit.apps.chat.engine.async_engine import ChatEngine
from conduit.apps.chat.ui.async_input import AsyncInput
from conduit.apps.chat.ui.enhanced_input import EnhancedInput
from conduit.apps.chat.ui.demo_tui import DemoInput
from conduit.apps.chat.ui.input_interface import InputInterface
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from rich.console import Console


def create_chat_app(
    preferred_model: str,
    welcome_message: str,
    system_message: str,
    input_mode: str,
    options: ConduitOptions,
) -> ChatApp:
    """
    Factory function to create a fully configured async ChatApp.
    """
    # Create dependencies
    params = GenerationParams(model=preferred_model, system=system_message)
    engine = ChatEngine(params=params, options=options)

    # Select and create input interface
    if input_mode == "demo":
        console = Console()
        input_interface: InputInterface = DemoInput(console)
    elif input_mode == "enhanced":
        console = Console()
        input_interface = EnhancedInput(console)
    else:
        input_interface = AsyncInput()

    # Create app with all dependencies
    app = ChatApp(
        engine=engine,
        input_interface=input_interface,
        welcome_message=welcome_message,
        verbosity=options.verbosity,
    )

    # Dependency Injection: Wire the engine back into the UI for history access
    if isinstance(input_interface, (EnhancedInput, DemoInput)):
        input_interface.set_engine(engine)

    return app
