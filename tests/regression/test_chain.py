from Chain.sync import Model, Response, Chain, Prompt
from Chain.batch import ModelAsync, AsyncChain
from Chain.model.clients.perplexity_client import PerplexityContent
from pytest import fixture


@fixture
def model_list() -> list[str]:
    return ["gpt3", "haiku", "gemini", "llama3.1:latest", "sonar"]


@fixture
def prompt_list() -> list[str]:
    return ["name ten mammals", "name ten birds", "name ten villains"]


# Straighforward text completion
# --------------------------------------


## Single sync call with each provider
def test_chain_with_single_sync_call():
    model = Model("gpt3")
    prompt = Prompt("Name ten mammals.")
    chain = Chain(model=model, prompt=prompt)
    response = chain.run()
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("haiku")
    prompt = Prompt("Name ten mammals.")
    chain = Chain(model=model, prompt=prompt)
    response = chain.run()
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("gemini")
    prompt = Prompt("Name ten mammals.")
    chain = Chain(model=model, prompt=prompt)
    response = chain.run()
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("llama3.1:latest")
    prompt = Prompt("Name ten mammals.")
    chain = Chain(model=model, prompt=prompt)
    response = chain.run()
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("sonar")
    prompt = Prompt("Name ten mammals.")
    chain = Chain(model=model, prompt=prompt)
    response = chain.run()
    assert isinstance(response, Response)
    assert isinstance(response.content, PerplexityContent)


def test_chain_with_async_calls(prompt_list):
    model = ModelAsync("gpt3")
    chain = AsyncChain(model=model)
    responses = chain.run(prompt_strings=prompt_list)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = ModelAsync("haiku")
    chain = AsyncChain(model=model)
    responses = chain.run(prompt_strings=prompt_list)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = ModelAsync("gemini")
    chain = AsyncChain(model=model)
    responses = chain.run(prompt_strings=prompt_list)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = ModelAsync("llama3.1:latest")
    chain = AsyncChain(model=model)
    responses = chain.run(prompt_strings=prompt_list)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = ModelAsync("sonar")
    chain = AsyncChain(model=model)
    responses = chain.run(prompt_strings=prompt_list)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])
    assert all(
        [isinstance(response.content, PerplexityContent) for response in responses]
    )
