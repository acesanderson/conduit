# Conduit

A Python framework for orchestrating LLM interactions across multiple providers with support for structured outputs, multimodal content, and async execution.

## Project Purpose

Conduit provides a unified interface for working with language models from OpenAI, Anthropic, Google, Perplexity, and Ollama. It handles provider-specific API differences, supports multimodal inputs (text, images, audio), enables structured outputs via Pydantic models, and provides built-in caching, progress tracking, and conversation history management. The framework supports both synchronous and asynchronous execution patterns for batch processing.

## Architecture Overview

### Core Modules

- **model/** - LLM provider abstraction and client management
  - `model.py` - Core Model class with lazy-loaded provider clients
  - `model_async.py` - Async variant for concurrent operations
  - `model_client.py` - Server-based model client for remote inference
  - `clients/` - Provider-specific implementations (OpenAI, Anthropic, Google, Ollama, Perplexity)
  - `models/` - Model registry, provider specifications, and capability definitions

- **conduit/** - Orchestration layer
  - `sync_conduit.py` - Synchronous prompt+model+parser chaining
  - `async_conduit.py` - Asynchronous batch execution with concurrency control

- **message/** - Multimodal content handling
  - `message.py` - Base message abstraction with provider serialization
  - `textmessage.py` - Standard text content
  - `imagemessage.py` - Image handling with base64 encoding and format conversion
  - `audiomessage.py` - Audio content with playback support
  - `messages.py` - List-like message collection with validation
  - `messagestore.py` - Persistent conversation history with TinyDB backend

- **request/** - Request construction and validation
  - `request.py` - Provider-agnostic request object with parameter validation
  - `clientparams.py` - Provider-specific parameter schemas
  - `outputtype.py` - Output type literals (text, image, audio)

- **result/** - Response handling
  - `response.py` - Successful response wrapper with usage statistics
  - `error.py` - Structured error information with stack traces
  - `result.py` - Union type for Response | ConduitError

- **prompt/** - Template management
  - `prompt.py` - Jinja2-based prompt templates with variable validation
  - `prompt_loader.py` - Lazy-loading registry for template files

- **parser/** - Structured output extraction
  - `parser.py` - Pydantic model wrapper for function calling and structured responses

- **progress/** - Progress tracking and display
  - `verbosity.py` - Configurable verbosity levels (SILENT to DEBUG)
  - `handlers.py` - Rich and plain text progress handlers
  - `tracker.py` - Event-based progress tracking for sync and async operations
  - `wrappers.py` - Decorator for automatic progress display
  - `display_mixins.py` - Rich and plain text formatting for Request/Response/Error objects

- **odometer/** - Token usage tracking
  - `Odometer.py` - Base odometer with aggregation by provider/model/date
  - `SessionOdometer.py` - In-memory session tracking
  - `ConversationOdometer.py` - Per-conversation usage tracking
  - `PersistentOdometer.py` - PostgreSQL-backed persistent storage
  - `OdometerRegistry.py` - Centralized odometer management with automatic persistence
  - `TokenEvent.py` - Token usage event structure
  - `database/` - Persistence backends (PostgreSQL, SQLite)

- **chat/** - Interactive chat interfaces
  - `chat.py` - Extensible CLI chat with command system

- **cli/** - Command-line interface framework
  - `cli.py` - Decorator-based CLI builder for custom chat applications

- **scripts/** - Entry point scripts
  - `chat_cli.py` - Interactive chat with model selection
  - `models_cli.py` - Model listing and inspection tool
  - `imagegen_cli.py` - Image generation CLI
  - `update_modelstore.py` - Model capability database updater
  - `update_ollama_list.py` - Ollama model registry updater

- **cache/** - Response caching (referenced but not in tree)
  - Likely `cache.py` - Content-addressable cache with TinyDB backend

## Dependencies

### External Libraries

**LLM Provider SDKs:**
- `anthropic` - Anthropic Claude API client
- `openai` - OpenAI GPT API client  
- `google-generativeai` - Google Gemini API client
- `ollama` - Local Ollama model runner

**Core Framework:**
- `instructor` - Structured output extraction from LLMs
- `pydantic` - Data validation and schema definition
- `jinja2` - Template rendering for prompts
- `tiktoken` - OpenAI tokenization utilities

**Storage:**
- `tinydb` - Lightweight JSON database for cache/history
- `psycopg2` / `dbclients` - PostgreSQL backend for odometer persistence

**CLI/Display:**
- `rich` - Terminal formatting and progress display
- `pydub` - Audio playback for AudioMessage
- `chafa` - Terminal image display (optional)

**Image Processing:**
- `pillow` (PIL) - Image format conversion and resizing
- `diffusers` - Local image generation (Flux models)

**Utilities:**
- `dotenv` - Environment variable management
- `pathlib` - Path handling

### Local Dependencies

None - standalone package.

## API Documentation

### Core Classes

#### Model
```python
Model(model: str, console: Optional[Console] = None)
```
Primary interface for LLM interactions. Automatically routes to appropriate provider client based on model name.

**Class Methods:**
- `models() -> dict` - List all available models by provider
- `validate_model(model: str) -> bool` - Check if model is supported
- `from_server(model: str) -> ModelClient` - Create server-based model client
- `stats()` - Display session token usage statistics

**Instance Methods:**
- `query(query_input, response_model=None, cache=True, temperature=None, verbose=Verbosity.PROGRESS, stream=False, max_tokens=None, output_type="text", index=0, total=0, request=None) -> Response | ConduitError`
- `tokenize(text: str) -> int` - Get token count for text

**Properties:**
- `console` - Effective Rich console (instance → Model class → Conduit class hierarchy)

#### ModelAsync
```python
ModelAsync(model: str, console: Optional[Console] = None)
```
Async variant of Model for concurrent operations. Same interface as Model but with async methods.

**Instance Methods:**
- `query_async(query_input, verbose=Verbosity.PROGRESS, response_model=None, cache=False, **kwargs) -> Response | ConduitError`

#### SyncConduit
```python
SyncConduit(model: Model, prompt: Prompt = None, parser: Parser = None)
```
Synchronous orchestration of prompt + model + parser pipeline.

**Class Attributes:**
- `message_store: MessageStore | None` - Shared message store for conversation history
- `console: Console | None` - Shared Rich console

**Methods:**
- `run(input_variables=None, messages=None, parser=None, verbose=Verbosity.PROGRESS, stream=False, cache=True, index=0, total=0, include_history=True, save=True) -> Response | ConduitError`

#### AsyncConduit
```python
AsyncConduit(model: ModelAsync, prompt: Prompt = None, parser: Parser = None)
```
Asynchronous batch execution with concurrent request handling and progress tracking.

**Methods:**
- `run(input_variables_list=None, prompt_strings=None, semaphore=None, cache=True, verbose=Verbosity.PROGRESS, print_response=False) -> list[Response]`

#### Request
```python
Request(model: str, messages: Messages | list[Message], temperature=None, stream=False, verbose=Verbosity.PROGRESS, response_model=None, max_tokens=None, output_type="text", client_params=None)
```
Provider-agnostic request object with automatic validation and serialization.

**Class Methods:**
- `from_query_input(query_input: str | Message | list[Message], **kwargs) -> Request`

**Instance Methods:**
- `generate_cache_key() -> str` - Generate deterministic cache key
- `to_cache_dict() -> dict` - Serialize for caching
- `from_cache_dict(cache_dict: dict) -> Request` - Deserialize from cache
- `to_openai() -> dict` - Convert to OpenAI API format
- `to_anthropic() -> dict` - Convert to Anthropic API format
- `to_google() -> dict` - Convert to Google API format
- `to_ollama() -> dict` - Convert to Ollama API format
- `to_perplexity() -> dict` - Convert to Perplexity API format

#### Response
```python
Response(message: Message, request: Request, duration: float, input_tokens: int, output_tokens: int, timestamp: str = now, emit_tokens: bool = True)
```
Successful LLM response with usage metadata and automatic token event emission.

**Properties:**
- `content` - Message content (str, Pydantic model, or list)
- `prompt` - Last user message
- `messages` - Full conversation (request + response)
- `total_tokens` - Combined token count
- `model` - Model identifier

**Methods:**
- `display()` - Display image if ImageMessage
- `play()` - Play audio if AudioMessage
- `to_cache_dict() -> dict`
- `from_cache_dict(cache_dict: dict) -> Response`

#### ConduitError
```python
ConduitError(info: ErrorInfo, detail: ErrorDetail = None)
```
Structured error with debugging information.

**Class Methods:**
- `from_exception(exc: Exception, code: str, category: str, **context) -> ConduitError`
- `simple(code: str, message: str, category: str) -> ConduitError`

**Nested Types:**
- `ErrorInfo(code, message, category, timestamp)` - Core error information
- `ErrorDetail(exception_type, stack_trace, raw_response, request_params, retry_count)` - Debug details

#### Parser
```python
Parser(pydantic_model: type[BaseModel])
```
Wrapper for Pydantic models to enable structured outputs.

**Class Attributes:**
- `_response_models: list[type[BaseModel]]` - Registry of all Pydantic models for deserialization

**Static Methods:**
- `to_perplexity(pydantic_model) -> type` - Convert wrapper models for Perplexity API
- `as_string(pydantic_model) -> str` - JSON schema string for caching

#### Prompt
```python
Prompt(prompt_string: str)
```
Jinja2 template wrapper with variable validation.

**Class Methods:**
- `from_file(filename: Path) -> Prompt`

**Instance Methods:**
- `render(input_variables: dict) -> str`

**Properties:**
- `input_schema: set` - Set of required template variables

#### MessageStore
```python
MessageStore(messages=None, console=None, history_file="", log_file="", pruning=False, auto_save=True)
```
Persistent conversation history with list-like interface and TinyDB backend.

**Instance Methods:**
- `add_new(role: str, content: str)` - Convenience method to create and add TextMessage
- `add_response(response: Response)` - Add user + assistant messages from Response
- `save()` / `load()` - Manual persistence control
- `view_history()` - Pretty-print conversation
- `clear()` - Remove all messages
- `query_failed()` - Remove last user message on error
- `get(index: int) -> Message | None` - Get by 1-based index
- `copy() -> MessageStore` - Create non-persistent copy
- `prune()` - Keep only last 20 messages
- All standard list methods (append, extend, insert, remove, pop, etc.)

**Properties:**
- `system_message: Message | None` - Get system message if present

### Message Types

#### Message (Base)
```python
Message(message_type: str, role: str, content: Any)
```
Abstract base with serialization and provider format conversion.

**Instance Methods:**
- `to_cache_dict() -> dict`
- `from_cache_dict(cache_dict: dict) -> Message`
- `to_openai() -> dict`
- `to_anthropic() -> dict`
- `to_google() -> dict`
- `to_ollama() -> dict`
- `to_perplexity() -> dict`

#### TextMessage
```python
TextMessage(role: str, content: str | BaseModel | list[BaseModel])
```
Standard text message with support for Pydantic objects.

#### ImageMessage
```python
ImageMessage.from_image_file(image_file: Path, text_content: str, role: str = "user") -> ImageMessage
ImageMessage.from_base64(image_content: str, text_content: str, mime_type: str = "image/png", role: str = "user") -> ImageMessage
```
Image message with automatic PNG conversion and resizing.

**Instance Methods:**
- `display()` - Display in terminal using chafa

**Properties:**
- `text_content: str` - Associated text prompt
- `image_content: str` - Base64-encoded PNG
- `mime_type: str` - Always "image/png" after processing

#### AudioMessage
```python
AudioMessage.from_audio_file(audio_file: Path, text_content: str, role: str = "user") -> AudioMessage
AudioMessage.from_base64(audio_content: str, text_content: str, format: str = "mp3", role: str = "user") -> AudioMessage
```
Audio message with playback support.

**Instance Methods:**
- `play()` - Play audio using pydub

**Properties:**
- `text_content: str` - Associated text prompt
- `audio_content: str` - Base64-encoded audio
- `format: str` - Audio format (mp3 or wav)

#### Messages
```python
Messages(messages: list[Message] = [])
```
List-like collection with turn order validation.

**Instance Methods:**
- `add_new(role: str, content: str)`
- `last() -> Message | None`
- `get_by_role(role: str) -> list[Message]`
- `user_messages() -> list[Message]`
- `assistant_messages() -> list[Message]`
- `system_messages() -> list[Message]`
- All standard list methods

**Properties:**
- `system_message: Message | None`

### Progress and Verbosity

#### Verbosity
```python
class Verbosity(Enum):
    SILENT = 0      # No output
    PROGRESS = 1    # Spinner/completion only
    SUMMARY = 2     # Basic request/response info
    DETAILED = 3    # Truncated messages in panels
    COMPLETE = 4    # Full messages in panels
    DEBUG = 5       # Full JSON with syntax highlighting
```

**Class Methods:**
- `from_input(value) -> Verbosity` - Convert bool/str/"v"/"vv"/etc. to Verbosity

### Odometer System

#### OdometerRegistry
```python
OdometerRegistry()
```
Centralized token tracking with automatic persistence.

**Attributes:**
- `session_odometer: SessionOdometer` - In-memory session tracking
- `conversation_odometers: dict[str, ConversationOdometer]` - Per-conversation tracking
- `persistent_odometer: PersistentOdometer` - PostgreSQL-backed storage

**Methods:**
- `register_conversation_odometer(conversation_id: str)`
- `emit_token_event(event: TokenEvent)` - Route event to all odometers

Access via `Model._odometer_registry` class attribute.

#### TokenEvent
```python
TokenEvent(provider: str, model: str, input_tokens: int, output_