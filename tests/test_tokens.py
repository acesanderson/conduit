from conduit.message.messagestore import MessageStore
from conduit.sync import Conduit, Prompt, Model, Verbosity, ConduitCache

VERBOSITY = Verbosity.PROGRESS
MESSAGE_STORE = MessageStore()
CACHE = ConduitCache()
Conduit.message_store = MESSAGE_STORE
Model.conduit_cache = CACHE

model = Model("haiku")
prompt = Prompt("Write a haiku about autumn leaves.")
conduit = Conduit(model=model, prompt=prompt)
response = conduit.run(verbose=VERBOSITY, index=1, total=5)

print(MESSAGE_STORE.__repr__())

prompt = Prompt("Explain the haiku in regular prose.")
conduit = Conduit(model=model, prompt=prompt)
response = conduit.run(verbose=VERBOSITY, index=2, total=5)

print(MESSAGE_STORE.__repr__())

prompt = Prompt("Summarize the explanation in one sentence.")
conduit = Conduit(model=model, prompt=prompt)
response = conduit.run(verbose=VERBOSITY, index=3, total=5)

print(MESSAGE_STORE.__repr__())

prompt = Prompt("Translate the summary into French.")
conduit = Conduit(model=model, prompt=prompt)
response = conduit.run(verbose=VERBOSITY, index=4, total=5)

print(MESSAGE_STORE.__repr__())

prompt = Prompt(
    "Write a 1,000 word essay on the philosophical ramifications of this conversation we've been having."
)
conduit = Conduit(model=model, prompt=prompt)
response = conduit.run(verbose=VERBOSITY, index=5, total=5)

print(MESSAGE_STORE.__repr__())
