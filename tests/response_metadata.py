from conduit.sync import Model, ConduitOptions
from conduit.config import settings

project_name = "test"
options = ConduitOptions(
    project_name=project_name,
    cache=settings.default_cache(project_name),
)
model = Model("gpt3", options=options)
response = model.query("What is the capital of France?")
print(f"Was this a cache hit? {response.metadata.cache_hit}")
