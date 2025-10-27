from conduit.sync import Model, Verbosity, ModelAsync, AsyncChain
from conduit.batch import ModelAsync, AsyncChain
from conduit.examples.fixtures import sample_prompt_strings


def test_plain_sync_response():
    model = Model("gpt3")
    for level in Verbosity:
        print(f"Testing Verbosity level: {level.name}")
        response = model.query("What is 2+2?", verbose=level)


def test_plain_sync_error():
    model = Model("gpt3")
    for level in Verbosity:
        print(f"Testing Verbosity level: {level.name}")
        error = model.query("What is 2+2?", verbose=level, return_error=True)


def test_plain_async_response():
    modelasync = ModelAsync("gpt3")
    chainasync = AsyncChain(model=modelasync)
    for level in Verbosity:
        print(f"Testing Verbosity level: {level.name}")
        responses = chainasync.run(prompt_strings=sample_prompt_strings)


def test_rich_sync_response():
    from rich.console import Console

    console = Console()
    Model._console = console
    # Tests
    model = Model("gpt3")
    for level in Verbosity:
        print(f"Testing Verbosity level: {level.name}")
        response = model.query("What is 2+2?", verbose=level)


def test_rich_sync_error():
    from rich.console import Console

    console = Console()
    Model._console = console
    model = Model("gpt3")
    for level in Verbosity:
        print(f"Testing Verbosity level: {level.name}")
        error = model.query("What is 2+2?", verbose=level, return_error=True)


def test_rich_async_response():
    from rich.console import Console

    console = Console()
    ModelAsync._console = console
    modelasync = ModelAsync("gpt3")
    chainasync = AsyncChain(model=modelasync)
    for level in Verbosity:
        print(f"Testing Verbosity level: {level.name}")
        responses = chainasync.run(prompt_strings=sample_prompt_strings)
