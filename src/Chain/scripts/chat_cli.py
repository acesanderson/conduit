from Chain.chat.chat import Chat
from Chain.chain.chain import Chain
from Chain.model.model import Model
from Chain.message.messagestore import MessageStore
from Chain.logs.logging_config import get_logger
from Chain.cache.cache import ChainCache
from pathlib import Path
from rich.console import Console
import readline
import argparse

# Constants
dir_path = Path(__file__).parent
_ = readline.get_current_history_length()  # Gaming the type hints.
logger = get_logger(__name__)  # Our logger
console = Console()
Chain._console = console
Model._console = console
Model._chain_cache = ChainCache(db_path=dir_path / ".chat_cache.db")  # Caching set up.
Chain._message_store = MessageStore(pruning=True)  # Non-persistant, but we should prune


def main():
    parser = argparse.ArgumentParser(description="Chat CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:latest",
        help="The model to use for chatting (default: llama3.1:latest)",
    )
    args = parser.parse_args()
    model_name = args.model
    # Validate model name
    if Model.validate_model(model_name) is False:
        console.print(f"[red]Invalid model name: {model_name}[/red]")
        return

    # Set the model based on command line argument
    c = Chat(Model(model_name))
    c.chat()


if __name__ == "__main__":
    main()
