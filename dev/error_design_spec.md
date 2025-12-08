Here is your design specification. Save this as `docs/design_specs/001_error_handling_architecture.md` or in your Obsidian vault under **Architecture \> Conduit**.

-----

# Design Spec: Conduit Error Handling & Logging Architecture

**Status:** Draft
**Context:** Refactoring from Pydantic-based error models to native Python Exception hierarchy.
**Goal:** Implement robust control flow, prevent circular dependencies, eliminate abstraction leaks, and separate "UI" from "Core".

## 1\. Core Philosophy

  * **Errors are Events, not Data:** We use Python `Exception` classes for control flow, not `pydantic.BaseModel`.
  * **The "Failure Interface":** The Exception hierarchy defines the API contract for failure. The UI/CLI should never need to import 3rd party libraries (like `openai` or `httpx`) to handle errors.
  * **Abstractions Must Hold:** Low-level SDK errors (e.g., `openai.RateLimitError`) must be caught at the boundary and re-raised as domain-specific Conduit errors.
  * **UI Agnostic:** Core logic raises exceptions with clean data. It never returns formatted strings (e.g., no `[red]Error...[/red]`). The UI layer decides how to render the error.

## 2\. Directory Structure (Package-Level Exceptions)

To respect Domain-Driven Design (DDD) while avoiding Python circular imports, exceptions are defined in dedicated `exceptions.py` files within their specific domains.

```text
src/conduit/
├── exceptions.py               # Root: ConduitError (Base)
├── core/
│   ├── model/
│   │   ├── exceptions.py       # ModelError, ContextWindowExceededError
│   │   ├── model_base.py
│   │   └── ...
│   └── clients/
│       ├── exceptions.py       # ProviderAPIError, RateLimitError
│       └── openai_client.py
├── storage/
│   ├── exceptions.py           # DatabaseError, CacheMissError
│   └── ...
└── apps/
    └── chat/
        └── engine/
            └── exceptions.py   # ConduitChatError (User-facing flow control)
```

## 3\. The Exception Hierarchy

### A. The Root (`src/conduit/exceptions.py`)

```python
class ConduitError(Exception):
    """Base exception for all Conduit logic. Catch this to handle ANY library failure."""
    pass
```

### B. Infrastructure/Client Layer (`src/conduit/core/clients/exceptions.py`)

```python
from conduit.exceptions import ConduitError

class ClientError(ConduitError):
    """Base for external API client failures."""
    pass

class ProviderAPIError(ClientError):
    """The provider (OpenAI/Anthropic) returned a 5xx error or is unreachable."""
    pass

class RateLimitError(ClientError):
    """We are being throttled (429)."""
    pass

class AuthError(ClientError):
    """Invalid API key or permission denied."""
    pass
```

### C. Domain Logic Layer (`src/conduit/core/model/exceptions.py`)

```python
from conduit.exceptions import ConduitError

class ModelError(ConduitError):
    """Base for inference/logic failures."""
    pass

class ContextWindowExceededError(ModelError):
    """Input tokens + Output tokens > Model Context Limit."""
    pass

class InvalidModelError(ModelError):
    """The requested model does not exist or is not supported."""
    pass
```

## 4\. Implementation Rules

### Rule 1: The Wrapper Pattern (`raise ... from`)

When catching a 3rd party error, **always** chain the exception to preserve the stack trace for debugging.

**Correct:**

```python
# src/conduit/core/clients/openai/client.py
from conduit.core.clients.exceptions import ProviderAPIError
import openai

try:
    response = self.client.chat.completions.create(...)
except openai.APIError as e:
    # Contextualize the error, but keep the original trace
    raise ProviderAPIError(f"OpenAI request failed: {e}") from e
```

### Rule 2: EAFP (Easier to Ask Forgiveness)

Do not check `if error: return`. Let Python bubble the error up.

  * **Core Logic:** Raises Exception.
  * **Middle Layer:** May catch specific exceptions to retry (e.g., `RateLimitError`).
  * **App/CLI Layer:** The *only* place that catches `ConduitError` to display a message to the user.

### Rule 3: The "No-Print" Policy

Core logic (`src/conduit/core/*`) must **NEVER** print to `stdout`.

  * **Log:** Use `logger.debug()` or `logger.warning()` for developer visibility.
  * **Raise:** Use Exceptions for control flow.
  * **Return:** Return values for success.

**Enforcement:**
Add `flake8-print` (`T201`) to `ruff` configuration in `pyproject.toml` to ban `print()` statements in core modules.

## 5\. Logging Architecture

### Centralized Setup

Setup logging **once** at the entry point (`main()` in CLI scripts).

**`src/conduit/logging.py`**

```python
import logging
from rich.logging import RichHandler

def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    # Silence noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

### Usage in Modules

```python
import logging
logger = logging.getLogger(__name__)

def complex_logic():
    logger.debug("Starting complex logic...")
    try:
        # ...
    except Exception:
        logger.error("Logic failed", exc_info=True)
        raise
```

## 6\. Migration Checklist

1.  [ ] **Create Exception Files:** Create the `exceptions.py` files in `root`, `core/model`, `core/clients`, and `storage`.
2.  [ ] **Update Clients:** Refactor `OpenAIClient`, `AnthropicClient`, etc., to try/except their SDK calls and raise `conduit.core.clients.exceptions`.
3.  [ ] **Refactor Model:** Ensure `Model` classes import these exceptions and allow them to bubble up (or handle retries).
4.  [ ] **Clean Core:** Grep for `print` and `return "[red]..."` in `src/conduit/core`. Replace with `logger` calls and `raise`.
5.  [ ] **Update UI:** Update `ChatApp` and `CLI` to catch `ConduitError` and print the formatted error message there.
6.  [ ] **Lock it down:** Enable `T201` (no-print) rule in `ruff`.
