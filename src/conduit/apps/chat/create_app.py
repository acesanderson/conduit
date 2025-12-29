"""
This module serves as the factory layer for the Conduit chat application, abstracting the dependency injection logic required to instantiate a fully configured `ChatApp`. It orchestrates the initialization of the `ChatEngine` for conversation state management and selects the appropriate `InputInterface` strategy (either standard asynchronous input or the enhanced `prompt_toolkit` UI) based on the provided mode.

The primary function connects generation parameters and runtime options to the engine while ensuring correct wiring between the input layer and the engine. This centralization allows distinct CLI entry points to share identical setup logic while supporting features like command auto-completion, which require circular dependency resolution between the UI and the engine.
"""

from conduit.apps.chat.app import ChatApp
from conduit.apps.chat.engine.async_engine import ChatEngine
from conduit.apps.chat.ui.async_input import AsyncInput
from conduit.apps.chat.ui.enhanced_input import EnhancedInput
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
    if input_mode == "enhanced":
        console = Console()
        input_interface: InputInterface = EnhancedInput(console)
    else:
        input_interface = AsyncInput()

    # Create app with all dependencies
    app = ChatApp(
        engine=engine,
        input_interface=input_interface,
        welcome_message=welcome_message,
        verbosity=options.verbosity,  # Verbosity comes from options now
    )

    if isinstance(input_interface, EnhancedInput):
        input_interface.set_engine(engine)

    return app
