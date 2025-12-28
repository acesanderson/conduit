## 2025-12-27

### 🔄 The Engine State Machine (FSM)

The new `Engine` in `core/engine.py` operates as a Finite State Machine. Instead of a linear script, it inspects the **tail** of the conversation history to decide what to do next.

#### 1. State Determination Logic

The state is derived dynamically from the `role` of the **last message** in the conversation (`conversation.last`):

| Last Message Role | Condition | Implied State | Engine Action |
| --- | --- | --- | --- |
| **`User`** | N/A | **`GENERATE`** | The user just spoke. We need the LLM to generate a response. |
| **`Tool`** | N/A | **`GENERATE`** | A tool just finished running. We need the LLM to see the result and synthesize an answer (or call another tool). |
| **`Assistant`** | `tool_calls` is present | **`EXECUTE`** | The LLM wants to act. The engine must run the requested tools. |
| **`Assistant`** | Text only | **`TERMINATE`** | The LLM has provided a final answer. The loop ends and control returns to the user. |
| **`System`** | (or empty) | **`INCOMPLETE`** | The conversation is not ready for processing. |

#### 2. The Execution Loop

The `Engine.run()` method executes a `while` loop that continues until the state becomes `TERMINATE` or a `max_steps` safety limit is reached.

1. **GENERATE Phase:**
* Calls `Model.query()`.
* Appends a new `AssistantMessage` to the conversation.
* *Loop repeats.*


2. **EXECUTE Phase:**
* Iterates through `tool_calls` in the last message.
* Invokes the corresponding capabilities.
* Appends `ToolMessage`s (results) to the conversation.
* *Loop repeats.* (This naturally transitions back to **GENERATE**, as the tail is now a `ToolMessage`).


3. **TERMINATE Phase:**
* Returns the final `Conversation` object to the caller.



---

### 📝 Amended Changelog: Conduit v1.1.0

Here is the summary of updates for your documentation or release notes.

#### 🚀 Architecture & Core

* **Finite State Machine (FSM):** Replaced linear "Chain" logic with a cyclic `Engine` that supports autonomous agentic loops (`Generate`  `Execute`).
* **Domain-Driven Design (DDD):** Refactored codebase into strict domains: `core` (kernel), `domain` (data), `clients` (I/O), `capabilities` (tools), and `storage`.
* **Strict Typing:** Replaced untyped dictionaries with Pydantic models for all internal data structures (`GenerationRequest`, `GenerationResponse`).
* **Discriminated Unions:** `Message` is now a strict union of `UserMessage`, `AssistantMessage`, and `ToolMessage`, replacing the generic `Message` class.

#### 🔌 Providers & Clients (ACL)

* **Anti-Corruption Layer:** Implemented a strict Adapter pattern. Each provider (OpenAI, Anthropic, Google, Ollama, Perplexity) now has a dedicated `Client` that converts internal DTOs to provider-specific `Payloads` at the boundary.
* **Remote Execution:** Added `RemoteClient` to offload inference to a `Headwater` server instance via `conduit --local`.
* **Unified Tokenization:** Standardized token counting across providers (using `tiktoken` fallback where native APIs are unavailable).

#### 🛠️ CLI & Applications

* **Click Rewrite:** Completely rewrote the CLI using `click` for better command handling and help generation.
* **New Commands:**
* `conduit query`: Main interface for LLM interaction (supports stdin piping).
* `conduit history`: View persisted conversation logs.
* `conduit wipe`: Clear conversation history.
* `conduit config`: Introspect current settings.


* **Aliases:** Added `ask` entry point as a shortcut for `conduit query`.

#### ⚙️ Orchestration & Middleware

* **Sync/Async Split:** Explicitly separated `ConduitSync` and `ConduitAsync` to handle environment setup before passing control to the `Engine`.
* **Middleware Context:** Introduced `middleware_context_manager` to handle side-effects (UI spinners, caching, odometer telemetry) uniformly across all model calls.

#### 🗑️ Breaking Changes

* **Removed:** `model.py` (Monolith), `sync_conduit.py`, `async_conduit.py`, `SessionOdometer.py` (merged into `OdometerRegistry`).
* **Renamed:** `tools/` is now `capabilities/tools/`.
* **Config:** `settings.toml` and XDG paths are now the single source of truth for configuration.
