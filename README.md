# Conduit

**The Universal Runtime for LLM Applications.**

Conduit is a Python framework for orchestrating Large Language Model interactions. It rejects the "Chain" metaphor in favor of a **Finite State Machine (FSM)** architecture, treating conversation history as state and prediction as a cyclic process.

It is built on three core principles:

1.  **Strict Typing:** Everything is a Pydantic model. No loose dictionaries.
2.  **No Magic:** Explicit dependency injection. You control the state.
3.  **Universal Loop:** A single runtime engine handles linear chat, RAG, and complex agentic loops using the same logic.

-----

## 📐 The Architecture

Legacy frameworks model LLM apps as "Chains" (DAGs). Conduit models them as a **State Machine**.

The core insight of Conduit is that all LLM interactions—whether a simple Q\&A or a multi-step autonomous agent—follow the same logic: **Predict the next message based on the current state.**


### The Components

1.  **The State (`Conversation`):** A passive, serializable container holding the message history.
2.  **The Engine (`Conduit`):** An active processor that inspects the *tail* of the conversation to determine the next state transition.
3.  **The Adapters (`Clients`):** An Anti-Corruption Layer (ACL) that normalizes disparate provider APIs (OpenAI, Anthropic, Ollama) into internal domain objects.
4.  **The Actors (`Capabilities`):** Executable units (Tools, Skills) that perform actions when the FSM enters the `EXECUTE` state.

-----

## 🗺️ Project Structure

Conduit is organized into clear domain boundaries to prevent circular dependencies and "God Objects."

```text
src/conduit/
├── core/                  # THE KERNEL
│   ├── engine.py          # The Universal Loop (FSM) logic
│   ├── prompt.py          # Jinja2 template management
│   └── parser.py          # Structured output parsing
│
├── domain/                # THE DATA (Pure Pydantic DTOs)
│   ├── conversation.py    # The State Container
│   ├── messages.py        # Discriminated Unions (User, Assistant, Tool)
│   └── request.py         # Internal Transport DTO
│
├── clients/               # THE I/O LAYER (Adapters & ACL)
│   ├── base.py            # Abstract Client Interface
│   ├── common.py          # Payload Type Definitions
│   ├── openai/            # Provider implementations...
│   ├── anthropic/
│   └── ...
│
├── capabilities/          # THE ACTORS
│   ├── executor.py        # Tool execution logic
│   ├── tools/             # Atomic functions (filesystem, search)
│   └── skills/            # Complex behaviors (personas, memories)
│
├── storage/               # PERSISTENCE
│   ├── repository.py      # Conversation/Message persistence
│   └── odometer/          # Token counting & Telemetry
│
└── apps/                  # CONSUMERS
    ├── cli/               # Command Line Interface
    └── chat/              # TUI Application
```

-----

## 🧠 Core Concepts

### 1\. The Message Union (`domain/messages.py`)

Conduit abandons inheritance for composition. Messages are a **Discriminated Union** of strict types.

  * **`UserMessage`**: Supports multimodal content blocks (Text + Image + Audio).
  * **`AssistantMessage`**: Atomic representation of a turn. Contains **Content** (final answer), **Reasoning** (hidden chain-of-thought), and **ToolCalls** (intent to act).
  * **`ToolMessage`**: The result of an execution, strictly linked to a call ID.

### 2\. The Universal Loop (`core/engine.py`)

There are no separate classes for "Agent" vs "Chat." There is only `Conduit.run()`.

The engine implements a standard Act-Observe-Think loop:

1.  **GENERATE:** If the tail is `UserMessage` or `ToolMessage` $\rightarrow$ Call LLM.
2.  **EXECUTE:** If the tail is `AssistantMessage` with `tool_calls` $\rightarrow$ Execute Tools.
3.  **TERMINATE:** If the tail is `AssistantMessage` with text only $\rightarrow$ Return to User.

### 3\. The Anti-Corruption Layer (`clients/`)

Conduit refuses to let provider idiosyncrasies leak into your business logic.

  * **Internal Domain:** We use `Conduit.Request` (generic).
  * **Provider Domain:** We define strict `Payload` models (e.g., `AnthropicPayload`, `OpenAIPayload`) that mirror the exact API spec of the vendor.
  * **The Adapter:** The `Client` class is responsible for converting `Request` $\rightarrow$ `Payload`.

-----

## 💻 Usage Examples

### 1\. The Direct Flow (Simple Chat)

For linear, synchronous interaction.

```python
from conduit.core.engine import Conduit
from conduit.domain.conversation import Conversation
from conduit.clients.openai.client import OpenAIClient

# 1. Initialize State
conv = Conversation()
conv.add_user_message("Why is the sky blue?")

# 2. Initialize Engine
client = OpenAIClient()
conduit = Conduit(client=client)

# 3. Run the Loop (Runs until TERMINATE state)
result = conduit.run(conv)
print(result.last_message.content)
```

### 2\. The Agentic Flow (Tools & Loops)

By simply adding tools, the Engine automatically switches to a cyclic FSM.

```python
from conduit.capabilities.tools import WeatherTool, StockTool
from conduit.clients.anthropic.client import AnthropicClient

# 1. Initialize State
conv = Conversation()
conv.add_user_message("What is the stock price of Apple compared to the temperature in NY?")

# 2. Initialize Engine with Capabilities
client = AnthropicClient()
tools = [WeatherTool(), StockTool()]
conduit = Conduit(client=client, tools=tools)

# 3. Run the Loop
# The Engine will:
#   1. GENERATE (Thought: I need stock price and weather)
#   2. EXECUTE (Runs both tools in parallel)
#   3. GENERATE (Synthesizes answer based on ToolMessages)
#   4. TERMINATE
final_conv = conduit.run(conv)
```

-----

## 🔮 Future Roadmap

  * **Remote Execution:** Because `Conversation` is pure data, the Engine can serialize the state, send it to a `Headwater` server, execute the heavy compute there, and return the mutated state.
  * **Branching:** Moving from a List-based history to a Tree-based history to support "regenerate" and "alternative timeline" features without data duplication.
  * **Telemetry:** The `Odometer` system will track token usage across the FSM lifecycle, attributing costs to specific states (Reasoning vs. Generation vs. Tooling).

