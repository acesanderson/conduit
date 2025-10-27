"""
Regression testing, behavior-driven.
Core APIs.
"""

from Chain.sync import Model, Response
from Chain.batch import ModelAsync, AsyncChain
from Chain.model.clients.perplexity_client import PerplexityContent
from pytest import fixture

models = ["gpt3", "haiku", "gemini", "llama3.1:latest", "sonar"]


@fixture
def prompt_list() -> list[str]:
    return ["name ten mammals", "name ten birds", "name ten villains"]


# Straighforward text completion
# --------------------------------------


# Single sync call with each provider
def test_model_single_sync_call():
    model = Model("gpt3")
    response = model.query("Hello, world!")
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("claude")
    response = model.query("Hello, world!")
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("llama3.1:latest")
    response = model.query("Hello, world!")
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("gemini")
    response = model.query("Hello, world!")
    assert isinstance(response, Response)
    assert isinstance(response.content, str)

    model = Model("sonar")
    response = model.query("Hello, world!")
    assert isinstance(response, Response)
    assert isinstance(response.content, PerplexityContent)


# Series of sync calls with each provider
def test_series_of_sync_calls(prompt_list):
    def loop_through_prompts(model: str):
        model_obj = Model(model)
        responses = []
        for prompt in prompt_list:
            responses.append(model_obj.query(prompt))
        return responses

    for model in models:
        responses = loop_through_prompts(model)
        assert isinstance(responses, list)
        assert all([isinstance(response, Response) for response in responses])


## Series of async calls with each provider
def test_series_of_async_calls(prompt_list):
    def async_through_prompts(model: str):
        model_obj = ModelAsync(model)
        chain = AsyncChain(model=model_obj)
        responses = chain.run(prompt_strings=prompt_list)
        return responses

    model = "gpt3"
    responses = async_through_prompts(model)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = "haiku"
    responses = async_through_prompts(model)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = "llama3.1:latest"
    responses = async_through_prompts(model)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])

    model = "gemini"
    responses = async_through_prompts(model)
    assert isinstance(responses, list)
    assert all([isinstance(response, Response) for response in responses])


# Structured responses
# --------------------------------------

## Single sync call with each provider

## Series of sync calls with each provider

## Series of async calls with each provider

# Audio message

## gpt

## local
