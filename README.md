# Conduit

Conduit is a unified framework for building multi-modal LLM applications. It provides a consistent interface across major providers (OpenAI, Anthropic, Google, Mistral, Perplexity, and Ollama) while handles the heavy lifting of persistence, caching, and structured data extraction.

## Quick Start

### Installation

```bash
pip install .
```

### Minimal Example

Execute a simple query using the synchronous interface:

```python
from conduit.sync import Conduit

# Initialize with a model and a prompt template
conduit = Conduit.create(
    model="gpt-4o",
    prompt="Tell me a fact about {{ topic }}."
)

# Run with template variables
response = conduit(topic="PostgreSQL")
print(response.content)
```

## Core Value Demonstration

Conduit excels at complex orchestration, such as extracting structured data using Pydantic models while enabling automatic caching and session persistence.

```python
from pydantic import BaseModel
from conduit.sync import Conduit
from conduit.utils.progress.verbosity import Verbosity

class ResearchSummary(BaseModel):
    key_findings: list[str]
    confidence_score: float

# Create a project-scoped conduit with persistence and caching enabled
conduit = Conduit.create(
    project_name="market-research",
    model="claude-3-5-sonnet",
    prompt="Analyze the following data: {{ data }}",
    output_type="structured_response",
    response_model=ResearchSummary,
    persist=True,  # Saves to Postgres session store
    cached=True,   # Uses Postgres cache to avoid redundant API calls
    verbosity=Verbosity.DETAILED
)

result = conduit(data="Recent trends in vector database adoption...")
print(f"Findings: {result.last.parsed.key_findings}")
```

## Functional Overview

### Supported Providers
Conduit unifies the following backends into a single API:
*   **Cloud**: OpenAI, Anthropic, Google Gemini, Mistral, Perplexity.
*   **Local**: Ollama (supports automated context window management and remote inference server routing).

### Key Components
*   **Models**: Stateless interfaces for generation. Use `ModelSync` or `ModelAsync` for direct interaction.
*   **Conduits**: High-level orchestrators that bind a Prompt, Model, and Options (caching/persistence) into a single callable.
*   **Conversations & Sessions**: A Directed Acyclic Graph (DAG) architecture for message history. Sessions allow for branching and resuming conversations across different execution runs.
*   **Strategies**: Pre-built complex workflows, such as Recursive Summarization, Map-Reduce, and Schema Extraction.

### Multi-Modal Support
Conduit handles images and audio through specialized namespaces:

```python
from conduit.sync import Model

model = Model("gpt-4o")

# Image analysis
response = model.image.analyze(
    prompt_str="What is in this image?",
    image="path/to/image.jpg"
)

# Audio generation (TTS)
audio_response = model.audio.generate(
    prompt_str="The quick brown fox jumps over the lazy dog."
)
audio_response.play()
```

## CLI Usage

Conduit includes a suite of command-line tools for interactive use and system administration.

| Command | Description |
| --- | --- |
| `ask "query"` | Quick persistent query to the default LLM. |
| `conduit chat` | Start an interactive REPL with tab-completion and multiline support. |
| `conduit batch` | Run multiple prompts in parallel from a file or stdin. |
| `conduit-dataset` | Manage and inspect evaluation datasets and gold standards. |
| `models` | List available models, providers, and their specific capabilities. |

## Persistence & Infrastructure

Conduit is designed to use PostgreSQL for enterprise-grade features:

1.  **Request Cache**: Deterministic caching based on a hash of the prompt, parameters, and message history.
2.  **Session Store**: Stores every turn of every conversation, scoped by project name.
3.  **Odometer**: A telemetry system that tracks token usage per model, provider, and host, including automatic rescue of unsaved events if the application crashes.

### Environment Configuration
Ensure your API keys are set in your environment:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
# For persistence/caching:
export POSTGRES_HOST="localhost"
```

## Advanced Summarization
Conduit includes specialized strategies for long-document processing:
*   **RecursiveSummarizer**: Automatically switches between one-shot and map-reduce based on the model's context window.
*   **HierarchicalTree**: Implements a bottom-up tree summarization (RAPTOR-lite) for massive corpora.
*   **ChainOfDensity**: Iteratively densifies summaries to increase information entropy.
