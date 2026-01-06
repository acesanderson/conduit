# Conduit Workflow Engine: Usage Guide

This guide outlines best practices for creating, composing, and executing workflows using the Conduit engine.

## Core Architecture

The system is built on four pillars:
1.  **Strategies:** Reusable, configurable units of business logic (e.g., `Summarizer`, `Translator`).
2.  **Workflows:** Orchestrators that chain multiple strategies together.
3.  **Steps:** The atomic functions decorated with `@step` that handle telemetry.
4.  **Harness:** The runtime container that manages configuration, tracing, and context.

---

## 1. Creating a Strategy

A **Strategy** is a class that implements a specific capability. To create a production-ready strategy, follow these rules:

1.  Inherit from the appropriate Protocol (e.g., `SummarizationStrategy`).
2.  Decorate your execution method (`__call__`) with `@step`.
3.  **Explicitly define** the parameters you use (for IDE support).
4.  **Always include `**kwargs`** (for infrastructure compatibility).
5.  Use `resolve_param` to handle configuration.

### Example: A Configurable Translator

```python
from conduit.core.workflow.step import step, resolve_param, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.core.model.model_async import ModelAsync

class TranslatorStrategy(SummarizationStrategy):
    """
    Translates text to a target language.
    """

    @step
    async def __call__(
        self, 
        text: str, 
        # 1. Explicitly define knobs you support
        target_language: str | None = None,
        model: str | None = None,
        # 2. Catch-all for infrastructure (run_id, trace context, etc.)
        **kwargs 
    ) -> str:
        # 3. Resolve Parameters (Runtime > Config > Default)
        language = resolve_param("target_language", "French", kwargs)
        model_name = resolve_param("model", "gpt-3.5-turbo", kwargs)

        # 4. Business Logic
        prompt = f"Translate the following to {language}:\n\n{text}"
        model_instance = ModelAsync(model_name)
        
        # Note: We pass **kwargs down just in case the model layer needs them
        result = await model_instance.query(prompt, **kwargs)
        
        # 5. Explicit Telemetry (Logic-specific)
        # We log the language because it's a key decision made inside this step.
        add_metadata("resolved_language", language)
        
        return result.content

```

---

## 2. Telemetry & Observability

Conduit uses a **Telemetric Middleware** to automatically capture infrastructure metrics, keeping your business logic clean.

### What is Captured IMPLICITLY? (Don't log these manually)

The framework automatically injects the following into your trace for every `@step`:

* **Inputs & Outputs:** The exact arguments passed to and returned from the function.
* **Latency:** Execution duration in seconds.
* **Status:** Success or Error state (including exception messages).
* **Token Usage:** `input_tokens` and `output_tokens` are summed up across **all** model calls made within the step.
* **Model Inventory:** `models_used` tracks which models were queried (e.g., `['gpt-4', 'gpt-3.5-turbo']`).
* **Config Resolution:** A record of every parameter resolved via `resolve_param` and its source.

### What should be added EXPLICITLY?

Use `add_metadata(key, value)` for **Domain-Specific Decisions** that the framework cannot guess.

* *Examples:* Number of chunks created, a calculated quality score, a branching decision ("took_fast_path"), or specific internal variables.

---

## 3. Composing Workflows

Workflows are simply Strategies that call other Strategies. Because `resolve_param` handles context automatically, you don't need to manually pass configuration objects around.

### Example: Translate-Then-Summarize

```python
class MultilingualSummaryWorkflow:
    def __init__(self):
        self.translator = TranslatorStrategy()
        self.summarizer = RecursiveSummarizer()

    @step
    async def __call__(self, text: str, **kwargs) -> str:
        # 1. First Step: Translate
        # We allow the Harness config to control the language, 
        # so we don't hardcode arguments here unless necessary.
        translated_text = await self.translator(text, **kwargs)

        # 2. Second Step: Summarize
        # We pass the kwargs through so the summarizer can receive 
        # its specific overrides if they exist.
        summary = await self.summarizer(translated_text, **kwargs)

        return summary

```

---

## 4. Running with the Harness

The `ConduitHarness` is your control plane.

### A. The "Production" Run (Config Injection)

Use this when running evaluations or production jobs where configuration is external (YAML, JSON, DB).

```python
from conduit.core.workflow.harness import ConduitHarness

# Configuration Dictionary
config = {
    "model": "gpt-3.5-turbo",
    "TranslatorStrategy.target_language": "German",
    "RecursiveSummarizer.chunk_ratio": 0.75
}

# 1. Instantiate Harness
harness = ConduitHarness(config=config)

# 2. Run Workflow
workflow = MultilingualSummaryWorkflow()
result = await harness.run(workflow, "Hello world, this is a test document.")

# 3. View Telemetry
harness.view_trace()

```

**Example Trace Output:**
Notice how `input_tokens` and `models_used` appear automatically without manual logging code.

```text
Step                           Duration   Status    Metadata                                         Output
------------------------------------------------------------------------------------------------------------
TranslatorStrategy             1.24s      success   input_tokens: 85, output_tokens: 42,             Hallo Welt...
                                                    models_used: ['gpt-3.5-turbo'],
                                                    resolved_language: German

RecursiveSummarizer            0.85s      success   input_tokens: 120, output_tokens: 60,            This is a summary...
                                                    models_used: ['gpt-3.5-turbo']

MultilingualSummaryWorkflow    2.10s      success   input_tokens: 205, output_tokens: 102            This is a summary...

```

### B. The "Drift Detection" Run

Use this in CI/CD or local debugging to ensure your configuration is actually being used.

```python
# Check for unused config keys (e.g. typos)
unused_keys = harness.report_unused_config()

if unused_keys:
    print(f"WARNING: The following config keys were ignored: {unused_keys}")
    # e.g. ["RecursiveSummerizer.model"] -> Typo in "Summarizer"!

```

### C. The "Discovery" Run (Schema Generation)

Use this to generate a template of all available settings for a new workflow.

```python
from conduit.core.workflow.harness import generate_config_schema

schema = await generate_config_schema(
    workflow_factory=MultilingualSummaryWorkflow,
    dummy_input="dummy text"
)

import json
print(json.dumps(schema, indent=2))

```

---

## Best Practices Checklist

1. **Defaults are code:** Define sensible defaults in your Python code (in `resolve_param` calls), not in `config` files.
2. **Explicit Arguments:** Always define named arguments for parameters you use. Use `None` as the default value to signal "check the config."
3. **Scope Names:** `resolve_param` automatically scopes using the class name. If you have two instances of the same class that need different configs, pass the parameters explicitly via `kwargs`.
4. **No Manual Token Counting:** Do not manually log `input_tokens` or `model` usage; the middleware handles this. Only log business logic metrics.
