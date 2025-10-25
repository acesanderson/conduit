from conduit.sync import (
    Model,
    Prompt,
    Conduit,
    Response,
    Verbosity,
    ConduitCache,
)
from conduit.parser.parser import Parser
from conduit.model.clients.perplexity_client import PerplexityContent
from conduit.batch import ModelAsync, AsyncConduit
from conduit.examples.sample_models import PydanticTestFrog

CACHE = ConduitCache()
PARSER = Parser(PydanticTestFrog)
STRUCTURED_PROMPT_STR = "Create an awesome frog."
UNSTRUCTURED_PROMPT_STR = "Tell me a joke."
UNSTRUCTURED_PROMPT_STRINGS = [
    "Tell me a joke.",
    "Tell me a funny story.",
    "Make me laugh.",
]
STRUCTURED_PROMPT_STRINGS = [
    "Create an awesome frog.",
    "Create a mean frog.",
    "Create a smart frog.",
]

PREFERRED_MODEL = "sonar"


# def test_perplexity_sync_unstructured():
#     model = Model(PREFERRED_MODEL)
#     prompt = Prompt(UNSTRUCTURED_PROMPT_STR)
#     conduit = Conduit(model=model, prompt=prompt)
#     response: Response = conduit.run(verbose=Verbosity.PROGRESS)
#     assert isinstance(response, Response)
#     assert isinstance(response.content, PerplexityContent)
#     print(response.content.text)
#     print(response.content.citations)


# def test_perplexity_sync_structured():
#     model = Model(PREFERRED_MODEL)
#     prompt = Prompt(UNSTRUCTURED_PROMPT_STR)
#     conduit = Conduit(model=model, prompt=prompt, parser=PARSER)
#     response: Response = conduit.run(verbose=Verbosity.PROGRESS)
#     assert isinstance(response, Response)
#     assert isinstance(response.content, PydanticTestFrog)
#     print(response.content.model_dump_json(indent=2))


# def test_perplexity_async_unstructured():
#     model = ModelAsync(PREFERRED_MODEL)
#     conduit = AsyncConduit(model=model)
#     response: Response = conduit.run(
#         prompt_strings=UNSTRUCTURED_PROMPT_STRINGS, verbose=Verbosity.PROGRESS
#     )
#     assert isinstance(response, list)
#     assert isinstance(response[0], Response)
#     for res in response:
#         assert isinstance(res.content, PerplexityContent)
#         print(res.content.text)
#         print(res.content.citations)


# def test_perplexity_async_structured():
#     model = ModelAsync(PREFERRED_MODEL)
#     conduit = AsyncConduit(model=model, parser=PARSER)
#     response: Response = conduit.run(
#         prompt_strings=STRUCTURED_PROMPT_STRINGS, verbose=Verbosity.PROGRESS
#     )
#     assert isinstance(response, list)
#     assert isinstance(response[0], Response)
#     for res in response:
#         assert isinstance(res.content, PydanticTestFrog)
#         print(res.content.model_dump_json(indent=2))
