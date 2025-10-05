from conduit.message.messagestore import MessageStore
from conduit.sync import Conduit, Prompt, Model, Response, Verbosity
from rich.console import Console
from pathlib import Path

console = Console()
verbose = Verbosity.COMPLETE
history_file = Path(__file__).parent / "history.json"

Model._console = console

m = MessageStore(history_file=history_file)

model = Model("gpt3")
response = model.query("name ten mammals", verbose=verbose)
m.extend(response.messages)
