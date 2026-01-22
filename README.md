# Conduit

Conduit is a unified framework for building LLM applications. It provides a single, high-level interface for interacting with multiple providers—including OpenAI, Anthropic, Google Gemini, Ollama, and Perplexity—while managing complex requirements like semantic caching, conversation persistence, and autonomous tool execution.

## Quick Start

### Installation

```bash
pip install conduit-project
```

### Basic Query

```python
from conduit.sync import Model

# Works with gpt-4o, claude-3-5-sonnet, gemini-1.5-pro, etc.
model = Model("gpt-4o")
response = model.query("Explain the significance of the year 1945.")

print(response.content)
print(f"Tokens used: {response.total_tokens}")
```

## Core Value Demonstration

Conduit excels at moving beyond simple text completion into structured workflows. It uses Pydantic to enforce data schemas and automatically handles tool execution loops.

### Structured Data Extraction

```python
from pydantic import BaseModel
from conduit.sync import Model

class ResearchSummary(BaseModel):
    key_entities: list[str]
    summary: str
    sentiment_score: float

model = Model("claude-3-5-sonnet")
response = model.query(
    "Analyze the latest news about fusion energy.",
    response_model=ResearchSummary
)

# response.message.parsed is a validated ResearchSummary instance
print(response.message.parsed.key_entities)
```

### Autonomous Tool Execution

Conduit can execute Python functions as tools, handling the multi-turn conversation loop automatically until the task is complete.

```python
from typing import Annotated
from conduit.sync import Conduit

async def get_weather(location: Annotated[str, "The city name"]) -> str:
    """Fetches current weather data."""
    return f"The weather in {location} is 22°C and sunny."

# Configure a conduit with a tool registry
conduit = Conduit.create(
    model="gpt-4o",
    prompt="Check the weather in London and San Francisco."
)
conduit.options.tool_registry.register_functions([get_weather])

result = conduit.run()
print(result.content)
```

## Features

*   **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google, Perplexity, and local Ollama instances.
*   **Semantic Caching**: Built-in Postgres-backed caching to prevent redundant API calls and reduce costs.
*   **Conversation Persistence**: Automatic DAG-based storage of conversation trees in Postgres, allowing for branching and resuming sessions.
*   **Multimodal Primitives**: Native support for image analysis/generation and audio transcription/TTS.
*   **Workflow Engine**: A state-machine based executor that manages tool calls and model reasoning steps.

## CLI Usage

Conduit includes several built-in CLI tools for interactive use and debugging.

### Interactive Chat
Launch a feature-rich terminal UI with persistent history and command completion:
```bash
chat
```

### Quick Query
Execute a one-off query directly from the terminal or via a pipe:
```bash
ask "What is the capital of France?"
cat file.py | ask "Refactor this code for readability"
```

### Model Management
List all supported models and their specific capabilities (context window, multimodal support, etc.):
```bash
models
```

## Configuration

Conduit uses environment variables for provider authentication.

| Environment Variable | Description |
|----------------------|-------------|
| `OPENAI_API_KEY` | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | Required for Claude models |
| `GOOGLE_API_KEY` | Required for Gemini models |
| `PERPLEXITY_API_KEY` | Required for Perplexity models |

For persistent storage and caching, ensure a Postgres instance is available. Conduit respects XDG base directory specifications for its local configuration and state files.

## Architecture Overview

Conduit follows a stateless "dumb pipe" philosophy for its core components:

1.  **Model**: A stateless interface to an LLM provider.
2.  **Conduit**: A template-aware orchestrator that renders prompts and manages context.
3.  **Engine**: A finite state machine that processes conversations, determining when to generate text and when to execute tools.
4.  **Middleware**: Handles cross-cutting concerns like UI spinners, logging, token usage tracking (Odometer), and caching.
