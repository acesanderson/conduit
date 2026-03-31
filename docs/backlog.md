# Conduit Backlog

## Model validation: local vs. server execution contexts

**Priority:** Medium
**Added:** 2026-03-31

### Background

Conduit was originally designed around a local Ollama instance. `ModelStore` maintains a registry of known models (`models.json` + optional `ollama_models.json` cache), and `validate_model()` raises `ValueError` for anything not in that registry. Two callsites enforce this at runtime:

- `ModelBase.__init__` — called whenever a `Model`/`ModelAsync` is instantiated
- `ModelStore.get_client()` — resolves which client class to use for a given model

This worked fine locally. It breaks when conduit runs **server-side** (on bywater/deepwater via HeadwaterServer), because the server receives a `BatchRequest` containing a model ID (e.g. `qwen3:8b`) that is valid for its local Ollama but not registered in the server's `ModelStore`.

### Current workaround

Both callsites now catch `ValueError` and pass through unknown model names, with `get_client()` defaulting unknown models to `OllamaClient`. This unblocks server-side inference but removes validation entirely.

### The real fix

Validation rules differ by execution context:

| Context | Validation needed |
|---|---|
| Local SDK execution | Strict — model must be in registry, correct client must be selected |
| Server-side execution (HeadwaterServer) | None — server owns its Ollama, it knows what it has |
| Remote execution via headwater client | Loose — model string passes through to the remote server |

The fix requires making execution context explicit at the `ModelBase` level, likely via `ConduitOptions` (which already carries `execution_mode`-adjacent fields) or a dedicated `ExecutionContext` enum. `get_client()` and `validate_model()` should branch on this rather than applying uniform local-registry validation.

### Related

- `conduit.domain.request.generation_params.GenerationParams._validate_model` — same catch+passthrough applied here
- `headwater_api.classes.conduit_classes.requests.BatchRequest` — the deserialization boundary where server-side validation currently triggers
- `conduit.core.clients.ollama.server_registry` — the per-server model cache (`ollama_models.json`) addresses the client-side awareness gap but not the server-side validation gap
