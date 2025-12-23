from conduit.sync import Model, Verbosity, GenerationRequest


model = Model.create("haiku", verbosity=Verbosity.SILENT)
assert model.options.verbosity == Verbosity.SILENT
# assert model.options.verbosity == Verbosity.COMPLETE
# response = model.query("what's up tigerlily?", verbosity=Verbosity.COMPLETE)
response = model.query("what's up tigerlily dude?")
print(response.request.options.verbosity)
