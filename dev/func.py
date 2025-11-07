from conduit.batch import ModelAsync, AsyncConduit, Response, Verbosity, Prompt

VERBOSITY = Verbosity.PROGRESS
MODEL_NAME = "llama3.1:latest"
PROMPT_STRINGS = """
Name thirteen blackbirds.
What is the capital of Lesotho?
What are career opportunities for assessment developers?
Name the top educational theorists of the 20th century.
""".strip().split("\n")

# Our conduit
model = ModelAsync(MODEL_NAME)
conduit = AsyncConduit(model=model)
