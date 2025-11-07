from rich.console import Console
from conduit.sync import ConduitCache, Model
from conduit.message.messagestore import MessageStore
from conduit.chat.chat_class import ConduitChat
from xdg_base_dirs import xdg_config_home


# Constants
PREFERRED_MODEL = "llama3.1:latest"
MESSAGE_STORE = MessageStore()
WELCOME_MESSAGE = "[green]Hello! Type /exit to exit.[/green]"
SYSTEM_MESSAGE = (
    (xdg_config_home() / "conduit" / "system_message.jinja2").read_text()
    if (xdg_config_home() / "conduit" / "system_message.jinja2").exists()
    else ""
)
CONSOLE = Console()
# Attach our singletons
CACHE = ConduitCache(name="conduit")
Model.conduit_cache = CACHE


def main():
    chat = ConduitChat(
        preferred_model=PREFERRED_MODEL,
        welcome_message=WELCOME_MESSAGE,
        system_message=SYSTEM_MESSAGE,
        message_store=MESSAGE_STORE,
        console=CONSOLE,
    )
    chat.chat()


if __name__ == "__main__":
    main()
