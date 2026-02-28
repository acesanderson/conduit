# Conduit CLI Health Check

**Date:** 2026-02-28

## Root cause: one systemic failure cascades everywhere

**`ModelStore.get_client()` hardcodes `model_list["openai"]` as the first lookup** (`modelstore.py:365`), but `ModelStore.models()` now returns `['anthropic', 'google', 'groq', 'perplexity', 'mistral', 'ollama']` — OpenAI was removed from the store but the routing logic wasn't updated. This `KeyError: 'openai'` blocks every command that tries to resolve a model client.

---

## Working

| CLI | Command | Result |
|-----|---------|--------|
| `models` | `models` | List all models by provider |
| `models` | `models -a` | Aliases dict |
| `models` | `models -e` | Embedding models list |
| `models` | `models -m <model>` | Full capability card |
| `models` | `models -m <badname>` | Fuzzy fallback ("Did you mean:") |
| `models` | `models -p <provider>` | Filter by provider (lowercase only) |
| `models` | `models -t <type>` | Filter by type |
| `tokens` | `tokens` | Full odometer: requests, tokens, by provider and model, today vs. all-time |
| `conduit` | `conduit history` | Session message history renders correctly |
| `conduit` | `conduit last` | Returns last message correctly |

---

## Broken — with root cause

**1. `ask` / `conduit query` — all invocations, all models**
```
KeyError: 'openai'  (modelstore.py:365)
```
`get_client()` does `if model_name in model_list["openai"]` before checking any other provider. Since `"openai"` is gone from the store, every query dies on the first line of client routing.

**2. `imagegen` — all subcommands including `history` and `last`**
```
KeyError: 'openai'  (modelstore.py:365)
```
`imagegen_cli.py:210` eagerly calls `ModelSync(model=args.model)` at parse time, before any subcommand dispatches. So even `imagegen history` (which should be read-only) hits this. The model init needs to be deferred to generation-only paths.

**3. `tokenize` — default and all models**
- Default: `gpt` alias → `gpt-5-mini` → `ValueError: Model not found locally: gpt-5-mini` (OpenAI removed, default alias is stale)
- Any other model: hits the same `KeyError: 'openai'` in `get_client()`

**4. `conduit config`**
```
KeyError: 'chat'  (base_commands.py:214)
```
`_build_cli()` in `cli_class.py` injects `ctx.obj` but never sets `ctx.obj["chat"]`. The `config` command expects it. One missing line in the DI block.

**5. `conduit get <N>`**
```
'function' object has no attribute 'messages'
```
`conduit last` (line 192) correctly calls `ctx.obj["conversation"]()` — note the `()`. `conduit get` (line 202) does `ctx.obj["conversation"]` without calling the lambda, passing the function itself to the handler instead of the Conversation object. One missing `()`.

**6. `conduit_cache` — all subcommands**
```
'AsyncPostgresCache' object has no attribute '_get_pool'
```
`cache_cli.py` calls `cache._get_pool()` directly, but that private method no longer exists on `AsyncPostgresCache` — presumably removed when `db_manager` was introduced to consolidate connection pooling.

---

## UX issues (not crashes)

| CLI | Issue |
|-----|-------|
| `models -p GOOGLE` | Fails with ValueError — display shows uppercase provider names, but validator requires lowercase. Mismatch. |
| `models -t chat` | Fails — `chat` is the natural guess but not a valid type. Valid types (`text_completion`, `reasoning`, `image_gen`, etc.) are non-obvious and not surfaced on error. |
| `tokenize` default | `gpt` alias resolves to `gpt-5-mini` which is gone. Needs a new default. |

---

## Not tested (interactive / destructive)

- `chat` — interactive TUI, can't test non-interactively
- `conduit wipe` — destructive, requires confirmation prompt
- `conduit_cache -w` — same

---

## Fix priority

| Priority | Location | Fix |
|----------|----------|-----|
| P0 | `modelstore.py:365` | Remove hardcoded `model_list["openai"]` check (or guard with `if "openai" in model_list`) |
| P0 | `imagegen_cli.py:210` | Defer `ModelSync` init to generate-only paths, not parse time |
| P1 | `base_commands.py:202` | `conduit get`: add missing `()` to `ctx.obj["conversation"]` |
| P1 | `cli_class.py` `_build_cli()` | Add `ctx.obj["chat"]` to the DI block |
| P1 | `cache_cli.py` | Update `_get_pool()` call to use `db_manager` API |
| P2 | `tokenize_cli.py` | Update default model alias to something that exists |
| P3 | `models_cli.py` | Normalize provider input to lowercase before validation |
