from Chain.model.model import Model
from Chain.request.request import Request
from Chain.result.response import Response

# OpenAI client request - leveraging unique OpenAI features
openai_request = {
    "max_tokens": 150,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.3,
    "stop": [".", "!", "?"],
}

# Anthropic client request - using Anthropic-specific sampling
anthropic_request = {
    "top_k": 40,
    "top_p": 0.9,
    "stop_sequences": ["Human:", "Assistant:"],
}

# Google client request - inherits OpenAI spec but with safety settings
google_request = {
    "max_tokens": 200,
    "presence_penalty": 0.1,
    "stop": ["\n\n", "END"],
}

# Perplexity client request - OpenAI spec focused on research/search
perplexity_request = {
    "max_tokens": 300,
    "frequency_penalty": 0.1,
}

# Ollama client request - extensive Ollama-specific options
ollama_request = {
    "num_ctx": 4096,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "stop": ["<|im_end|>", "\n\n"],
}


def test_openai_request():
    model = Model("gpt-3.5-turbo-0125")
    request = Request.from_query_input(
        query_input="name ten mammals",
        model="gpt-3.5-turbo-0125",
        client_request=openai_request,
    )
    response = model.query(request=request)
    assert isinstance(response, Response)


def test_anthropic_request():
    model = Model("claude-3-5-haiku-20241022")
    request = Request.from_query_input(
        query_input="name ten mammals",
        model="claude-3-5-haiku-20241022",
        client_request=anthropic_request,
    )
    response = model.query(request=request)
    assert isinstance(response, Response)


def test_google_request():
    model = Model("gemini-1.5-flash")
    request = Request.from_query_input(
        query_input="name ten mammals",
        model="gemini-1.5-flash",
        client_request=google_request,
    )
    response = model.query(request=request)
    assert isinstance(response, Response)


def test_perplexity_request():
    model = Model("sonar")
    request = Request.from_query_input(
        query_input="name ten mammals", model="sonar", client_request=perplexity_request
    )
    response = model.query(request=request)
    assert isinstance(response, Response)


def test_ollama_request():
    model = Model("llama3.1:latest")
    request = Request.from_query_input(
        query_input="name ten mammals",
        model="llama3.1:latest",
        client_request=ollama_request,
    )
    response = model.query(request=request)
    assert isinstance(response, Response)
