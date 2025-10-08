## Conduit

A Python library for orchestrating synchronous and asynchronous interactions with Large Language Models. Conduit provides a unified interface for multiple providers, simplifying common tasks like templating, conversation management, structured data extraction, and batch processing.

### Core Features

- **Unified API:** Interact with models from OpenAI, Anthropic, Google, Ollama, and Perplexity through a single consistent interface.
- **Synchronous & Asynchronous Execution:** Use Conduit for single, blocking queries and AsyncConduit for high-throughput batch processing.
- **Structured Data Extraction:** Define Pydantic models and use the Parser to reliably extract structured JSON from model outputs.
- **Conversation Management:** Automatically manage chat history with MessageStore, including optional persistence to a file.
- **Multimodal Support:** Natively handle ImageMessage and AudioMessage for vision and audio transcription/generation tasks.
- **Usage Tracking:** Monitor token usage with Odometer, which tracks costs by provider, model, and session, with support for a persistent Postgres backend.
- **Powerful Templating:** Leverage Jinja2 for dynamic prompt creation using the Prompt class.
- **Informative Logging:** Control console output with a tiered Verbosity system, offering everything from a silent mode to full debug logs with rich formatting.

### Quick Start
#### Installation
```bash
# (Assuming standard setup.py or pyproject.toml)
pip install .
Set your API keys in a .env file in your project root:

Code snippet

OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
PERPLEXITY_API_KEY="pplx-..."
```
#### Synchronous Query

This is the most common use case for a single LLM call.

```python
from conduit.sync import Conduit, Model, Prompt

# 1. Define a model and a prompt template
model = Model("gpt-4o-mini")
prompt = Prompt("Explain {{topic}} in simple terms for a beginner.")

# 2. Create and run the Conduit
conduit = Conduit(model=model, prompt=prompt)
response = conduit.run(input_variables={"topic": "quantum computing"})

print(response.content)
Asynchronous (Batch) Queries
Process multiple prompts concurrently for high throughput.
```

```python
from conduit.batch import AsyncConduit, ModelAsync

# 1. Define an async model and a list of prompts
model = ModelAsync("claude-3-5-sonnet")
prompts = [
    "What are the three laws of thermodynamics?",
    "Who wrote 'The Hitchhiker's Guide to the Galaxy'?",
    "What is the capital of Nepal?"
]

# 2. Create and run the AsyncConduit
conduit = AsyncConduit(model=model)
responses = conduit.run(prompt_strings=prompts)

for resp in responses:
    print(f"- {resp.content}\n")
```
### Key Components

- **Conduit / AsyncConduit:** The primary orchestrators. They combine a Model, Prompt, and optional Parser to execute LLM queries.
- **Model / ModelAsync:** The interface to LLM providers. It handles API requests, caching, and tokenization.
- **Prompt:** A wrapper around Jinja2 templates for dynamic and reusable prompt engineering.
- **Message & MessageStore:** Classes for representing conversation turns (user, assistant, system). MessageStore adds persistence for chat history.
- **Parser:** Integrates with Pydantic to enforce structured, validated JSON output from models.

### Advanced Usage
#### Structured Output with Pydantic

Extract structured data from unstructured text by providing a Pydantic model.

```python
from conduit.sync import Conduit, Model, Prompt
from conduit.parser import Parser
from pydantic import BaseModel, Field

class Character(BaseModel):
    name: str = Field(description="The character's full name")
    role: str = Field(description="The character's primary role in the story")

# 1. Define the model, prompt, and a parser with your Pydantic model
model = Model("claude-3-5-sonnet")
prompt = Prompt("Describe the protagonist of the novel 'Dune'.")
parser = Parser(Character)

# 2. Run the conduit with the parser
conduit = Conduit(model=model, prompt=prompt, parser=parser)
response = conduit.run()

# The response content is now a validated Pydantic object
character: Character = response.content
print(f"Name: {character.name}, Role: {character.role}")
```

#### Conversational Chat

Use a MessageStore to automatically track and include conversation history in subsequent requests.

```python
from conduit.sync import Conduit, Model, Prompt
from conduit.message import MessageStore

# 1. Initialize a MessageStore for history
message_store = MessageStore(history_file="chat_history.json")
Conduit.message_store = message_store # Assign to the class to be used by all instances

model = Model("gpt-4o-mini")

# 2. Run sequential prompts; history is managed automatically
print("--- First turn ---")
prompt1 = Prompt("Name three famous physicists from the 20th century.")
conduit1 = Conduit(model=model, prompt=prompt1)
conduit1.run()

print("\n--- Second turn ---")
prompt2 = Prompt("Which one of them is most famous for their work on relativity?")
conduit2 = Conduit(model=model, prompt=prompt2)
response = conduit2.run()

print(f"\nModel Response:\n{response.content}")

# You can view the history at any time
print("\n--- Conversation History ---")
message_store.view_history()
```
### Command-Line Tools
Conduit includes several scripts for quick interaction from your terminal.
#### Interactive Chat: Start a chat session with any supported model.

```bash
python -m conduit.scripts.chat_cli --model "llama3.1:latest"
```
#### Model Store: List available models and view their capabilities.

```bash
# List all models by provider
python -m conduit.scripts.models_cli

# List models from a specific provider
python -m conduit.scripts.models_cli -p openai

# Get details for a specific model
python -m conduit.scripts.models_cli -m "claude-3-5-sonnet"
```
