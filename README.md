# Chain

A comprehensive Python framework for building LLM applications with support for multiple providers, multimodal content, structured outputs, and advanced features like caching, progress tracking, and token usage monitoring.

## Features

### 🤖 **Multi-Provider Support**
- **OpenAI** (GPT-4, GPT-3.5, DALL-E, Whisper)
- **Anthropic** (Claude 3.5 Sonnet/Haiku)
- **Google** (Gemini 2.5, Imagen)
- **Perplexity** (Research models with citations)
- **Ollama** (Local models)
- **Hugging Face** (Custom models)

### 🎯 **Structured Outputs**
- Pydantic model validation and parsing
- Type-safe responses with automatic validation
- JSON schema generation and enforcement

### 🎨 **Multimodal Support**
- **Images**: Analysis, generation, display
- **Audio**: TTS, transcription, playback
- **Video**: Analysis capabilities
- Support for multiple formats with automatic conversion

### ⚡ **Performance & Reliability**
- Async/await support for concurrent operations
- Intelligent caching with SQLite/PostgreSQL backends
- Token usage tracking and cost monitoring
- Progress indicators with rich console output

### 🔧 **Developer Experience**
- CLI chat interface with extensible commands
- Rich error handling and debugging
- Comprehensive logging and verbosity levels
- Hot-swappable models and providers

## Quick Start

### Basic Usage

```python
from Chain import Chain, Model, Prompt

# Simple query
model = Model("gpt-4o")
response = model.query("What is the meaning of life?")
print(response.content)

# Template-based prompts
prompt = Prompt("Explain {{topic}} in {{style}} terms")
chain = Chain(model=model, prompt=prompt)

response = chain.run(input_variables={
    "topic": "quantum computing", 
    "style": "simple"
})
print(response.content)
```

### Structured Outputs

```python
from Chain import Chain, Model, Parser
from pydantic import BaseModel

class Animal(BaseModel):
    name: str
    species: str
    habitat: str
    diet: list[str]

model = Model("claude-3-5-sonnet")
parser = Parser(Animal)
chain = Chain(model=model, parser=parser)

response = chain.run("Tell me about a lion")
animal = response.content  # Returns Animal instance
print(f"{animal.name} is a {animal.species}")
```

### Multimodal Examples

#### Image Analysis
```python
from Chain import Model
from Chain.message.imagemessage import ImageMessage

model = Model("gpt-4o")  # Vision-capable model
image_msg = ImageMessage.from_image_file(
    image_file="photo.jpg",
    text_content="What's in this image?"
)

response = model.query(image_msg)
print(response.content)
```

#### Image Generation
```python
model = Model("dall-e-3")
response = model.query(
    "A cyberpunk cityscape at sunset",
    output_type="image"
)
response.display()  # Shows image in terminal
```

#### Audio Processing
```python
from Chain.message.audiomessage import AudioMessage

model = Model("gpt-4o-audio-preview")
audio_msg = AudioMessage.from_audio_file(
    audio_file="recording.mp3",
    text_content="Transcribe this audio"
)

response = model.query(audio_msg)
print(response.content)
```

### Async Operations

```python
from Chain import AsyncChain, ModelAsync

model = ModelAsync("gpt-4o")
chain = AsyncChain(model=model)

# Process multiple prompts concurrently
prompts = [
    "Explain photosynthesis",
    "What is machine learning?", 
    "Describe the water cycle"
]

responses = chain.run(prompt_strings=prompts)
for response in responses:
    print(response.content)
```

### Chat Interface

```python
from Chain import Chat, Model
from Chain.message.messagestore import MessageStore

# Persistent chat with history
messagestore = MessageStore(history_file="chat_history.json")
model = Model("claude-3-5-haiku")
chat = Chat(model=model, messagestore=messagestore)

chat.chat()  # Starts interactive CLI
```

## Advanced Features

### Caching

```python
from Chain import Model, ChainCache

# Enable caching
Model._chain_cache = ChainCache(db_path="cache.db")

model = Model("gpt-4o")
response = model.query("Expensive query", cache=True)
# Subsequent identical queries return cached results
```

### Token Tracking

```python
from Chain import Model

model = Model("gpt-4o")
response = model.query("Hello world")

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Duration: {response.duration:.2f}s")

# View session statistics
Model.stats()
```

### Progress & Verbosity

```python
from Chain import Model, Verbosity

model = Model("gpt-4o")

# Different verbosity levels
response = model.query("Query", verbose=Verbosity.SILENT)    # No output
response = model.query("Query", verbose=Verbosity.PROGRESS)  # Spinner only
response = model.query("Query", verbose=Verbosity.DETAILED)  # Full details
response = model.query("Query", verbose=Verbosity.DEBUG)     # JSON dump
```

### Provider-Specific Parameters

```python
from Chain import Model
from Chain.request.request import Request

# OpenAI-specific parameters
model = Model("gpt-4o")
request = Request.from_query_input(
    query_input="Be creative",
    model="gpt-4o",
    temperature=0.9,
    client_params={
        "max_tokens": 1000,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3
    }
)
response = model.query(request=request)
```

### LLM Decorator

```python
from Chain.llm_decorator import llm

@llm(model="claude-3-5-haiku")
def summarize_text(text: str, length: str):
    """
    Summarize the following text in {{length}} format:
    
    {{text}}
    """

summary = summarize_text(
    text="Long article content...", 
    length="bullet points"
)
print(summary)
```

## Installation

```bash
pip install chain-llm  # When published
# or for development:
git clone https://github.com/yourusername/chain
cd chain
pip install -e .
```

### Environment Setup

```bash
# Required API keys (set as needed)
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
export PERPLEXITY_API_KEY="your_key_here"
```

## CLI Tools

### Chat Interface
```bash
python -m Chain.scripts.chat_cli --model claude-3-5-haiku
```

### Model Management
```bash
# List all models
python -m Chain.scripts.models_cli

# Get model details
python -m Chain.scripts.models_cli -m gpt-4o

# Filter by provider
python -m Chain.scripts.models_cli -p anthropic
```

### Image Generation
```bash
python -m Chain.scripts.imagegen --model dall-e-3 "A beautiful landscape"
```

## Architecture

### Core Components

- **Model**: Interface to LLM providers with unified API
- **Chain**: Orchestrates model, prompt, and parser
- **Request**: Encapsulates all parameters for LLM calls
- **Response**: Structured response with metadata
- **Messages**: Conversation management with validation
- **MessageStore**: Persistent conversation storage

### Message Types

- **TextMessage**: Standard text content
- **ImageMessage**: Images with analysis/generation
- **AudioMessage**: Audio content and transcription
- **VideoMessage**: Video analysis (planned)

### Progress System

- **Verbosity**: Configurable output levels
- **Handlers**: Rich console or plain text progress
- **Tracking**: Individual and concurrent operation monitoring

### Provider Architecture

Each provider implements a common interface:
- **Client**: Provider-specific API interaction
- **Parameters**: Validation for provider capabilities
- **Conversion**: Request/response format adaptation

## Error Handling

```python
from Chain import Model
from Chain.result.error import ChainError

model = Model("gpt-4o")
result = model.query("Test query")

if isinstance(result, ChainError):
    print(f"Error: {result.info.message}")
    print(f"Category: {result.info.category}")
    if result.detail:
        print(f"Stack trace: {result.detail.stack_trace}")
else:
    print(f"Success: {result.content}")
```

## Contributing

1. **Model Support**: Add new providers in `Chain/model/clients/`
2. **Message Types**: Extend multimodal support in `Chain/message/`
3. **Features**: Core functionality in respective modules
4. **Tests**: Add regression tests in `Chain/tests/`

### Development Setup

```bash
git clone https://github.com/yourusername/chain
cd chain
pip install -e .[dev]
pytest  # Run tests
```

## Roadmap

- [ ] Video message support
- [ ] WebSocket streaming
- [ ] Distributed computing
- [ ] Plugin system
- [ ] GUI interface
- [ ] Workflow orchestration (ChainML)
