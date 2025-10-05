from conduit.message.messagestore import MessageStore
from conduit.sync import Conduit, Model, Prompt, Verbosity, ConduitCache
from pathlib import Path
from rich.console import Console

console = Console()
verbosity = Verbosity.PROGRESS

history_file = Path(__file__).parent / "test_history.json"

ms = MessageStore(history_file=history_file, console=console)
Conduit._message_store = ms
Model._conduit_cache = ConduitCache()
Model._console = console

# Stage one of conversation
prompt = Prompt("name ten mammals")
speaker1 = Model("gpt3")
conduit = Conduit(model=speaker1, prompt=prompt)
response1 = conduit.run(verbose=verbosity)
print(response1)

# Stage two of conversation
prompt = Prompt("which, if any, are endangered?")
conduit = Conduit(model=speaker1, prompt=prompt)
response2 = conduit.run(verbose=verbosity)
print(response2)

# Stage three of conversation
prompt = Prompt("Are any of these aquatic?")
conduit = Conduit(model=speaker1, prompt=prompt)
response3 = conduit.run(verbose=verbosity)
print(response3)
