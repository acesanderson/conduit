from conduit.sync import Conduit, Verbosity

m = Conduit.create(
    "gpt3",
    # "name the most dangerous animals",
    "tell me more about that saltwater one",
    persist="test",
    # system="speak like Shakespeare",
    verbosity=Verbosity.DEBUG,
)
# m.options.cache.wipe()
r = m.run()
print(r)
