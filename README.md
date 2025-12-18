# Conduit

> A lightweight, unified Python framework for building LLM applications

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Conduit provides a clean, consistent interface for working with language models from multiple providers (OpenAI, Anthropic, Google, Perplexity, Ollama). It handles provider-specific API differences, supports multimodal inputs, enables structured outputs, and includes built-in caching, progress tracking, and conversation management.

## ✨ Key Features

- **🔌 Multi-Provider Support** - Unified interface for OpenAI, Anthropic, Google, Perplexity, and Ollama
- **📊 Structured Outputs** - Extract type-safe data with Pydantic models
- **🎭 Multimodal Content** - Handle text, images, and audio seamlessly
- **⚡ Async/Sync Execution** - Efficient batch processing with concurrency control
- **💾 Built-in Caching** - Content-addressable response caching
- **📝 Conversation History** - Persistent message storage with TinyDB
- **📊 Token Tracking** - Comprehensive usage monitoring and analytics
- **🎨 Rich Progress Display** - Beautiful terminal output with Rich
- **🔧 Template System** - Jinja2-based prompt templates

## 🚀 Quick Start

### Installation

```bash
pip install conduit
```

### Basic Usage

```python
from conduit import Model

# Simple query
model = Model("gpt-4o")
response = model.query("Explain quantum computing in one sentence")
print(response.content)
```

### Structured Outputs

```python
from pydantic import BaseModel
from conduit import Model, Parser

class Animal(BaseModel):
    name: str
    species: str
    habitat: str

model = Model("gpt-4o")
parser = Parser(Animal)

response = model.query(
    "Tell me about a red panda",
    response_model=parser
)

animal = response.content  # Type-safe Animal object
print(f"{animal.name} is a {animal.species}")
```

### Async Batch Processing

```python
import asyncio
from conduit import ModelAsync, AsyncConduit, Prompt

async def process_batch():
    model = ModelAsync("gpt-4o-mini")
    prompt = Prompt("Summarize: {{ text }}")
    conduit = AsyncConduit(model, prompt)
    
    inputs = [
        {"text": "Article 1..."},
        {"text": "Article 2..."},
        {"text": "Article 3..."}
    ]
    
    responses = await conduit.run(input_variables_list=inputs)
    return [r.content for r in responses]

results = asyncio.run(process_batch())
```

### Multimodal Inputs

```python
from pathlib import Path
from conduit import Model, ImageMessage

model = Model("gpt-4o")
image_msg = ImageMessage.from_image_file(
    Path("chart.png"),
    "What trends do you see in this chart?"
)

response = model.query(image_msg)
print(response.content)
```

## 📚 Core Concepts

### Model

The primary interface for LLM interactions. Automatically routes requests to the appropriate provider.

```python
model = Model("gpt-4o")  # OpenAI
model = Model("claude-3-5-sonnet-20241022")  # Anthropic
model = Model("gemini-2.0-flash-exp")  # Google
model = Model("llama3.2")  # Ollama
```

### Conduit

Orchestrates prompt templates, models, and parsers into reusable pipelines.

```python
from conduit import SyncConduit, Model, Prompt, Parser

model = Model("gpt-4o")
prompt = Prompt("Analyze {{ data }} for {{ metric }}")
parser = Parser(AnalysisResult)

conduit = SyncConduit(model, prompt, parser)
response = conduit.run({"data": "...", "metric": "accuracy"})
```

### Messages

Flexible message types for different content:

```python
from conduit import TextMessage, ImageMessage, AudioMessage

text = TextMessage(role="user", content="Hello!")
image = ImageMessage.from_image_file(Path("img.png"), "Describe this")
audio = AudioMessage.from_audio_file(Path("voice.mp3"), "Transcribe")
```

### Conversation History

Persistent conversation storage:

```python
from conduit import MessageStore, SyncConduit

store = MessageStore(history_file="conversation.json")
SyncConduit.message_store = store

# Conversations are automatically saved and restored
```

## 🛠️ CLI Tools

Conduit includes several command-line utilities:

```bash
# Interactive chat
chat

# List available models
models

# Generate images
imagegen "a serene mountain landscape"

# View token usage
tokens

# Update model registry
update
```

## 🎯 Advanced Features

### Custom Verbosity Levels

```python
from conduit import Verbosity

model.query("...", verbose=Verbosity.DEBUG)  # Full request/response JSON
model.query("...", verbose=Verbosity.SILENT)  # No output
```

### Token Usage Tracking

```python
# View session statistics
Model.stats()

# Access token events
from conduit.odometer import OdometerRegistry
registry = Model._odometer_registry
print(registry.session_odometer.get_summary())
```

### Response Caching

```python
# Cache enabled by default
response1 = model.query("What is AI?", cache=True)
response2 = model.query("What is AI?", cache=True)  # Instant, from cache
```

### Progress Display

```python
# Automatic progress tracking for batch operations
async_conduit.run(
    input_variables_list=large_dataset,
    verbose=Verbosity.PROGRESS  # Shows progress bar
)
```

## 📖 Documentation

- [API Reference](docs/api_reference.md) - Comprehensive API documentation
- [Cookbook](docs/cookbook.md) - Recipes and patterns
- [Building a Chatbot](docs/building_a_simple_chatbot.md) - Tutorial

## 🏗️ Architecture

```
conduit/
├── model/          # Provider abstraction and clients
├── conduit/        # Orchestration layer (sync/async)
├── message/        # Multimodal message types
├── request/        # Request construction and validation
├── result/         # Response and error handling
├── prompt/         # Jinja2 template system
├── parser/         # Structured output extraction
├── progress/       # Progress tracking and display
├── odometer/       # Token usage monitoring
├── chat/           # Interactive chat interfaces
└── cli/            # Command-line tools
```

## 🔧 Configuration

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

Or use a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [instructor](https://github.com/jxnl/instructor) - Structured outputs
- [pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [rich](https://github.com/Textualize/rich) - Terminal formatting
- Provider SDKs: OpenAI, Anthropic, Google, Ollama