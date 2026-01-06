"""
OVERVIEW
--------
Workflow separates the DEFINITION of a workflow (code) from its OBSERVATION and
CONFIGURATION (runtime). It uses a "Harness" pattern to inject state via
ContextVars, keeping the domain logic pure.

CORE CONCEPTS
-------------
1. WORKFLOW (The Orchestrator)
   - A callable that defines the sequence of operations (A -> B -> C).
   - It is purely functional and stateless regarding infrastructure.

2. STEP (The Unit of Work)
   - A function decorated with `@step`.
   - It performs actual logic (LLM calls, data processing).
   - It automatically logs its inputs/outputs/latency to the Trace.
   - It pulls configuration via `resolve_param()` for runtime tuning.

3. STRATEGY (The Interchangeable Unit)
   - A pre-baked Step (often a class) that wraps a specific behavior behind a common interface.
   - Designed to be configurable at runtime, supporting immediate instantiation with defaults,
     specific one-off overrides, or harness-injected configuration.

4. HARNESS (The Runtime)
   - Wraps the execution of a Workflow.
   - Manages the lifecycle of Trace and Config context variables.
   - Acts as a "Configuration Scanner" to detect available tunable parameters.

TUNING & NAMESPACING
--------------------
Configuration is resolved using a strict precedence rule via `resolve_param()`:

   1. RUNTIME: Explicit kwargs passed to the function (e.g., `await step(model="gpt-4")`).
   2. SCOPED:  Harness config "{scope}.{key}"       (e.g., "RecursiveSummarizer.model").
   3. GLOBAL:  Harness config "{key}"               (e.g., "model").
   4. DEFAULT: The hardcoded fallback provided in the code.

This architecture supports three distinct lifecycles:
   - Immediate: Run with defaults (`Strategy()("text")`)
   - Specific:  Run with overrides (`Strategy()("text", model="gpt-4")`)
   - Pipeline:  Run with injected config (`Harness(config={...}).run(Strategy(), "text")`)
"""
