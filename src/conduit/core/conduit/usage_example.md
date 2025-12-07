Conduit is a level of abstraction on top of Model, essentially.
Conduits create a Conversation object to feed to the Engine FSM class.

SyncConduit:
```python
model = Model("gpt3")
prompt = Prompt("Tell me a joke about {topic}. Then design a Frog.")
parser = Parser(Frog)
system = SystemMessage("You are a helpful assistant.")
conduit = Conduit(
    model=model,
    prompt=prompt,
    parser=parser,
    system=system,
    # Conduit params
    verbosity=Verbosity.COMPLETE,
    cache="frog",
    persist=True,
    # Generation Params (kwargs) -- note, special handling for "model"
    temperature=0.7,
response = conduit.run(input_variables={"topic": "computers"})
)
```
Injected under the hood:
- Conversation
- Params (assemble from init kwargs)
- ConversationStore


### Other stuff
- convsersation store is immutable snapshot of conversation at a point in time.
- delete ConduitError. Use standard exceptions and custom exceptions by domain.
- properties: cache, conversation, store, params
