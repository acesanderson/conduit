# Conduit

Conduit is a lightweight, unified framework for building multimodal LLM applications in Python. It provides a standardized interface for interacting with multiple providers, managing conversation state via a Directed Acyclic Graph (DAG) architecture, and orchestrating complex workflows with built-in telemetry and caching.

## Quick Start

### Installation

Conduit requires Python 3.12 or higher.

```bash
pip install .
```

### Minimal Example

Generate a response from any supported provider using the synchronous interface.

```python
from conduit.sync import Model

# Initialize a model (OpenAI, Anthropic, Gemini, Ollama, etc.)
model = Model("gpt-4o")

# Simple text query
response = model.query("Explain quantum entanglement in one sentence.")
print(response.content)
```

## Core Value Demonstration

This example demonstrates multimodal input, structured data extraction using Pydantic, and persistent caching in a single workflow.

```python
from pydantic import BaseModel
from conduit.sync import Conduit, Prompt, ConduitOptions

class Analysis(BaseModel):
    summary: str
    key_entities: list[str]
    sentiment: float

# Define a prompt template with Jinja2 syntax
prompt = Prompt("""
Analyze the following document and return a structured response.
<document>
{{ text }}
</document>
""")

# Configure orchestration options
options = ConduitOptions(
    project_name="document_audit",
    use_cache=True,  # Enable Postgres-backed caching
)

# Execute the conduit with structured output
conduit = Conduit.create(
    model="claude-3-5-sonnet",
    prompt=prompt,
    options=options,
    response_model=Analysis,
    output_type="structured_response"
)

result = conduit.run(input_variables={"text": "Large corpus of text..."})
print(result.message.parsed.summary)
```

## Architecture Overview

Conduit is organized into four logical layers:

| Layer | Component | Responsibility |
| :--- | :--- | :--- |
| **Primitives** | `Message`, `Conversation` | State management and message threading using a DAG structure. |
| **Execution** | `Model` | Stateless interface for model interaction (Text, Image, Audio). |
| **Orchestration** | `Conduit` | Ties a `Prompt` to a `Model` with specific `Options` (caching, tools). |
| **Pipeline** | `Workflow` | Telemetry-aware execution tree using the `@step` decorator. |

### Supported Providers
The framework abstracts provider-specific SDKs into a unified payload format.
*   **Cloud:** OpenAI, Anthropic, Google Gemini, Perplexity, Mistral.
*   **Local:** Ollama.
*   **Remote:** Headwater/Siphon server clusters.

## Basic Usage

### Batch Processing
Process multiple inputs concurrently with built-in rate limiting and progress tracking.

```python
from conduit.batch import ConduitBatchSync

batch = ConduitBatchSync.create(
    model="gpt-4o-mini",
    prompt="Translate this to French: {{text}}",
    max_concurrent=10
)

inputs = [{"text": "Hello"}, {"text": "Goodbye"}]
results = batch.run(input_variables_list=inputs)
```

### Multimodal Operations
Models support native image and audio modalities through specialized namespaces.

```python
from conduit.sync import Model

model = Model("gpt-4o")

# Image Analysis
analysis = model.image.analyze(
    prompt_str="What is in this image?",
    image="path/to/image.png"
)

# Text-to-Speech
audio = model.audio.generate(
    prompt_str="This is a test of the emergency broadcast system."
)
audio.save_image("output.mp3") # Saves generated binary
```

### Telemetry and Caching
Conduit includes a persistent "Odometer" to track token usage across models and providers, stored in a local Postgres instance.

| Feature | Description |
| :--- | :--- |
| **Postgres Cache** | Semantic and exact-match caching for LLM responses to reduce latency and cost. |
| **Odometer** | Real-time tracking of input/output tokens per model, provider, and project. |
| **Trace Logging** | Complete execution logs for every `@step` in a workflow, including durations and metadata. |

## Command Line Interface

Conduit provides several CLI entry points for interaction and system management:

*   `conduit query "prompt"`: Quick one-off queries.
*   `ask "prompt"`: Shortcut for piped LLM operations.
*   `chat`: Enter an interactive REPL session with persistent history.
*   `tokens`: View aggregate token usage and costs.
*   `models`: List and fuzzy-search available model specifications.
*   `imagegen`: CLI for DALL-E and Gemini image generation.
