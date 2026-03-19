# Developer's Guide: Writing Conduit Strategies

A strategy is an async callable that takes a structured input and a config dict, runs some LLM workflow, and returns a string. The eval framework runs strategies across many input/config combinations, so keeping strategies self-contained and stateless is important.

---

## Anatomy of a Strategy

```python
from __future__ import annotations

from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.core.workflow.step import step, add_metadata


class MyStrategy(SummarizationStrategy):

    # 1. Declare your config schema
    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        temperature: float | None = None
        # ... all params your strategy reads

    config_model = Config  # exposes schema to the eval framework

    # 2. Implement __call__
    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)   # validate + apply defaults

        # ... your logic here

        add_metadata("some_metric", value)  # optional trace metadata
        return result
```

That's the whole pattern. Three moving parts: `Config`, `config_model`, and `__call__`.

---

## The Config Class

Every strategy defines an inner `Config(BaseModel)` with all parameters it reads.

- **All fields should have defaults.** A config with no overrides should produce reasonable behavior.
- **Use `extra="ignore"`** so passing a broader config dict (e.g. one with keys for a parent strategy or Chunker) doesn't raise.
- **Set `config_model = Config`** at class level. This is what `validate_config()` in the eval framework uses to surface mismatches before a run.
- Instantiate once at the top of `__call__`: `cfg = self.Config(**config)`. From that point, read from `cfg`, not from the raw dict.

Common shared fields:

| Field | Type | Default | Used by |
|---|---|---|---|
| `model` | `str` | `"gpt-oss:latest"` | Any LLM call |
| `temperature` | `float \| None` | `None` | `GenerationParams` |
| `max_tokens` | `int \| None` | `None` | `GenerationParams` |
| `top_p` | `float \| None` | `None` | `GenerationParams` |
| `project_name` | `str` | `"conduit"` | `ConduitOptions` |
| `chunk_size` | `int` | `12000` | `Chunker` |
| `overlap` | `int` | `500` | `Chunker` |

If your strategy delegates to `Chunker`, include `chunk_size` and `overlap` in your `Config` even though you don't read them directly — they'll flow through when you pass `config` to `Chunker`.

---

## The `@step` Decorator

Decorate `__call__` with `@step`. It:

- Records step name, inputs, output, duration, and status to the trace (when one is active — the eval framework sets this up automatically via `run_eval`)
- Provides a slot for `add_metadata()` calls within the step

You don't manage tracing manually. Just decorate and call `add_metadata()` for anything useful to capture.

```python
add_metadata("num_chunks", len(chunks))
add_metadata("input_tokens", response.metadata.input_tokens)
```

---

## Composing Strategies

Pass `config` through unchanged when calling sub-strategies. They'll parse out what they need via their own `Config`.

```python
# Delegate to a sub-strategy
result = await OneShotSummarizer()(input, config)

# Override a specific key for a sub-call
result = await MapReduceSummarizer()(input, {**config, "chunk_size": effective_threshold})
```

Chunker follows the same pattern:

```python
chunker = Chunker()
chunks = await chunker(text, config)  # Chunker reads chunk_size, overlap from config
```

---

## What NOT to Do

```python
# Don't touch the context vars — the eval framework owns these
context.config.set(config)       # NO
context.use_defaults.set(True)   # NO

# Don't use get_param — read from cfg directly
model = get_param("model", default="gpt3")   # NO
model = cfg.model                            # YES
```

---

## Eval Framework Integration

The eval framework (`evals/evals.py`) calls your strategy like this:

```python
warnings = validate_config(strategy, config)   # checks config against Config model
result = await strategy(input, config)         # runs with trace active
```

Validation is informational — mismatches are collected in `RunResult.warnings`, not raised as exceptions. But they will surface if your config is missing required fields or has type errors.

To run your strategy through the eval framework, just add it to `run.py`:

```python
from conduit.strategies.summarize.summarizers.my_strategy import MyStrategy

configs = [{"model": "gpt-oss:latest", "my_param": "value"}]
runs = await generate_runs(inputs=ds, configs=configs, strategy=MyStrategy())
```
