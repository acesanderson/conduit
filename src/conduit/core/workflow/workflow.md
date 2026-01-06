# Conduit Workflow Engine

## Overview
Workflow separates the DEFINITION of a workflow (code) from its OBSERVATION and CONFIGURATION (runtime). It uses a "Harness" pattern to inject state via ContextVars, keeping the domain logic pure and allowing for powerful features like auto-tracing and configuration injection.

## Core Concepts

1. **WORKFLOW** (The Orchestrator)
   - A callable that defines the sequence of operations (A -> B -> C).
   - It is purely functional and stateless regarding infrastructure.

2. **STEP** (The Unit of Work)
   - A function decorated with `@step`.
   - It performs actual logic (LLM calls, data processing).
   - It automatically logs its inputs/outputs/latency to the Trace.
   - It pulls configuration via `resolve_param()` for runtime tuning.

3. **STRATEGY** (The Interchangeable Unit)
   - A pre-baked Step (often a class) that wraps a specific behavior behind a common interface.
   - Designed to be configurable at runtime, supporting explicit overrides, harness injection, or defaults.

4. **HARNESS** (The Runtime)
   - Wraps the execution of a Workflow.
   - Manages the lifecycle of Trace and Config context variables.
   - Acts as a "Configuration Scanner" to detect available tunable parameters.

## Tuning & Namespacing

Configuration is resolved using a strict **4-Layer Cascade** via `resolve_param()`:

1.  **RUNTIME OVERRIDE (High Priority)**
    * Explicit arguments passed to the function call.
    * *Example:* `await strategy(text, model="gpt-4")`
    * *Note:* If the argument is `None`, it falls through to the next layer.

2.  **SCOPED CONFIGURATION**
    * Harness config key matching `{Scope}.{Key}`.
    * *Example:* `{"RecursiveSummarizer.model": "gpt-4"}`

3.  **GLOBAL CONFIGURATION**
    * Harness config key matching just `{Key}`.
    * *Example:* `{"model": "gpt-3.5-turbo"}`

4.  **DEFAULT (Low Priority)**
    * The hardcoded fallback provided in the code.
    * *Example:* `resolve_param("model", "gpt-3.5")`

## Developer Experience

### Auto-Binding Magic
You do not need to manually parse `kwargs`. The `@step` decorator automatically captures all arguments (positional and keyword) and makes them available to `resolve_param`.

**Usage Pattern:**
```python
@step
async def summarize(text: str, model: str | None = None) -> str:
    # 1. Tries to use the 'model' arg passed above (if not None).
    # 2. Checks config for 'Summarizer.model'.
    # 3. Checks config for 'model'.
    # 4. Defaults to "gpt-3.5".
    final_model = resolve_param("model", "gpt-3.5")
