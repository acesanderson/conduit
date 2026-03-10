# llama.cpp Server Integration — Design Spec

**Status**: Early design
**Date**: 2026-03-09
**Motivation**: Ollama does not expose GBNF grammar, logit bias, or raw completion prompts through any of its API endpoints. llama.cpp's native `/completion` endpoint supports all three. This integration gives conduit a "bare metal" inference path for use cases that require it.

---

## Network Topology

```
Laptop (bianders-mn7180)
    |
    +-- WireGuard VPN (172.16.0.x) or LAN (10.0.0.x)
    |
    +-> Caruana (172.16.0.4 / 10.0.0.82)   <- Postgres, MongoDB, etc.
    |
    +-> AlphaBlue (172.16.0.2 / 10.0.0.87) <- headwater (Ollama + conduit server)
                                             <- llama-server [NEW, fixed port 8090]
```

AlphaBlue already hosts:
- **Ollama** (port 11434)
- **headwater** (conduit HTTP server, wraps Ollama)

Adding:
- **llama-server** (port 8090, one instance, one model loaded at startup)

Host resolution for the llama.cpp endpoint reuses the existing `siphon_server`
address from `dbclients.discovery.host.get_network_context()` — same machine,
different port.

---

## Server: llama.cpp on AlphaBlue

### Installation

Download a pre-built CUDA binary from the llama.cpp GitHub releases page.
Pick the ubuntu-x64-cuda variant — it links against the same CUDA stack Ollama
already uses. No compilation needed.

```bash
wget <release-url>
unzip -d ~/llama.cpp/ llama-*.zip
```

### Startup Command

```bash
~/llama.cpp/llama-server \
  --model ~/.ollama/models/blobs/<gguf-hash> \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
  --port 8090 \
  --host 0.0.0.0
```

Key flags:
- `--n-gpu-layers 99` — push all layers to GPU; adjust down if VRAM is tight
- `--ctx-size` — set explicitly; llama.cpp default (512) is too small
- `--host 0.0.0.0` — required for remote access from the laptop
- One model per server instance; model selection is a startup decision, not per-request

### Model Strategy

Single dedicated instance for grammar/logprob work. Reuse the GGUF file Ollama
already has on disk. Locate it:

```bash
ollama show --verbose <model-name>
# "model" field under Parameters points to the blob hash file
```

Add a shell alias on AlphaBlue for the startup command so it's a one-liner to
spin up when needed.

### VRAM Consideration (open question)

Running Ollama and llama-server simultaneously loading the same model doubles
VRAM consumption. Options:
1. Unload the Ollama model first (`ollama stop <model>`) before starting llama-server
2. Use a smaller quantization in llama-server (e.g., Q4_K_M vs Q8)
3. Accept that llama-server is only active during classifier workloads, not alongside Ollama

Resolution depends on AlphaBlue's actual VRAM headroom. TBD.

---

## dbclients-project Changes

**File**: `src/dbclients/discovery/host.py`

Add a `get_llamacpp_url()` helper. No changes to the `NetworkContext` dataclass —
the host is already captured in `siphon_server`.

```python
LLAMACPP_PORT = 8090

def get_llamacpp_url() -> str:
    """
    Returns the base URL for the llama.cpp server running on AlphaBlue.
    Host resolved the same way as siphon_server (same machine, different port).
    """
    ctx = get_network_context()
    return f"http://{ctx.siphon_server}:{LLAMACPP_PORT}"
```

---

## conduit-project Changes

### Design decision: thin module, not a Client subclass

llama.cpp is not a general-purpose LLM provider. The `/completion` endpoint has
a fundamentally different interface from chat completions: it takes a raw prompt
string, not a messages list. There is no conversation history, no tool calling,
no structured output. Forcing it into the `Client` subclass hierarchy would mean
implementing inapplicable methods and distorting the abstraction.

Instead: a standalone module at `src/conduit/llamacpp/`, mirroring the
`conduit/embeddings/` pattern — direct async httpx calls, no class hierarchy.

### Module layout

```
src/conduit/llamacpp/
    __init__.py
    client.py       # LlamaCppClient -- thin async httpx wrapper
    payload.py      # LlamaCppPayload (Pydantic model for /completion request)
    response.py     # LlamaCppResponse (Pydantic model for /completion response)
```

### Payload

Maps to the `/completion` request body:

```python
class LlamaCppPayload(BaseModel):
    prompt: str
    grammar: str | None = None           # GBNF grammar string
    logit_bias: list[list] | None = None # [[token_id, bias], ...]
    n_probs: int = 0                     # top-N token logprobs to return
    n_predict: int = 1                   # max tokens to generate (-1 = unlimited)
    temperature: float = 0.0
    seed: int = -1
    stop: list[str] | None = None
```

### Response

The `/completion` endpoint returns:

```json
{
  "content": "1",
  "tokens_evaluated": 42,
  "tokens_predicted": 1,
  "logprobs": [
    {"token": "1", "logprob": -0.0002, "bytes": [49]}
  ]
}
```

```python
class LlamaCppTokenLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int]

class LlamaCppResponse(BaseModel):
    content: str
    tokens_evaluated: int
    tokens_predicted: int
    logprobs: list[LlamaCppTokenLogprob] = []
```

### Client

```python
class LlamaCppClient:
    """
    Thin async client for the llama.cpp /completion endpoint.
    Focused on grammar-constrained generation and logprob retrieval.
    """

    def __init__(self, base_url: str | None = None):
        from dbclients.discovery.host import get_llamacpp_url
        self.base_url = base_url or get_llamacpp_url()

    async def complete(self, payload: LlamaCppPayload) -> LlamaCppResponse: ...
    async def tokenize(self, content: str) -> list[int]: ...  # /tokenize -- for resolving logit_bias token IDs
    async def ping(self) -> bool: ...  # /health
```

### Caller pattern

```python
client = LlamaCppClient()
response = await client.complete(LlamaCppPayload(
    prompt="input: eagle::output: 0:::input: whale::output: 1:::input: beaver::output:",
    grammar='root ::= ("0" | "1")',
    n_probs=5,
    n_predict=1,
    temperature=0.0,
))
confidence = math.exp(response.logprobs[0].logprob)
```

---

## Capabilities This Unlocks (priority order)

| Capability | Why it needs llama.cpp |
|---|---|
| GBNF grammar | Not exposed by Ollama at any endpoint |
| Logit bias | Not exposed by Ollama |
| Raw completion prompts | Ollama applies chat template; /completion takes raw string |
| LoRA adapter hot-swap | Future: fine-tuned classifier checkpoint eval |
| Speculative decoding | Future: long-form generation throughput |
| KV cache slot reuse | Future: shared-prefix batching for classifier workloads |

---

## What This Does NOT Do

- No model management (Ollama owns that; reuse its GGUF files)
- No message history, tool calling, or structured output
- No integration with conduit's `Client` class hierarchy or `OutputType` dispatch
- No headwater involvement — direct laptop -> AlphaBlue:8090 connection

---

## Open Questions

1. **VRAM headroom**: Can Ollama and llama-server coexist on AlphaBlue, or does
   one need to yield? Check with `nvidia-smi` after loading both.

2. **Which model by default?** Likely the same model used for Ollama classifier
   work. Confirm the GGUF blob path.

3. **Startup automation**: Manual SSH + alias for now. Headwater management
   endpoint is a future option if llama-server becomes a persistent fixture.
