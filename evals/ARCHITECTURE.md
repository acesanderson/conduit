# Evals Architecture

## Three Layers

| Layer | Directory | Purpose |
|---|---|---|
| **Scaffolding** | `evals/` | Reusable infrastructure: `EvalRunner`, `ConduitDatasetAsync`, `make_gemini_judge`, `load_golden_dataset`, `persist` |
| **Abstractions** | `evals/` | Reusable eval types: one file per eval methodology, e.g. `effective_context_window.py` |
| **Jobs** | `jobs/` | Runnable entry points: one file per job, thin callers that wire an abstraction to specific models/servers |

Imports go downward only: `jobs/` → `evals/` abstractions → `evals/` scaffolding.

---

## How to Add a New Eval

Use `evals/effective_context_window.py` + `jobs/effective_context_window.py` as the canonical example.

**1. Write the abstraction in `evals/<eval_name>.py`**

Subclass `EvalRunner` and override `publish()`. Put any pure helper functions (binning, scoring, formatting) in the same file.

```python
# evals/my_eval.py
from __future__ import annotations
from runner import EvalRunner, EVAL_FUNCTION

class MyEvalRunner(EvalRunner):
    async def publish(self, ds, doc_meta):
        eval_results = await ds.evals.list(eval_function=EVAL_FUNCTION)
        # analyse and print results
```

**2. Write tests in `evals/tests/test_<eval_name>.py`**

Test pure functions (binning logic, scoring helpers, etc.) — not network calls. `sys.path.insert` points to `evals/`:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from my_eval import my_pure_function
```

**3. Write the job entry point in `jobs/<eval_name>.py`**

Just `main()` — parse args, build `run_matrix`, instantiate your runner, call `asyncio.run(runner.run(...))`.

```python
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
from my_eval import MyEvalRunner
```

Standard CLI flags: `--dry-run`, `--cron`, `--limit`, `--project`. Add any job-specific flags (e.g. `--model`, `--server`).

**4. Publish to Cronicle**

Deploy first:
```bash
bash scripts/deploy.sh alphablue
```

Then create a Cronicle event via the API (one event per model or configuration):

```bash
curl -s -X POST http://172.16.0.2:3012/api/app/create_event/v1 \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "my eval - modelname",
    "enabled": 1,
    "plugin": "shellplug",
    "target": "alphablue",
    "timing": {},
    "max_children": 1,
    "timeout": 0,
    "catch_up": 0,
    "params": {
      "script": "#!/bin/bash\nset -eo pipefail\nsource /home/fishhouses/.secrets\nsource /home/fishhouses/.exports\nexport XDG_DATA_HOME=/home/fishhouses/.local/share\nexport XDG_CONFIG_HOME=/home/fishhouses/.config\nexport XDG_STATE_HOME=/home/fishhouses/.local/state\ncd /home/fishhouses/Brian_Code/conduit-project\nexec /home/fishhouses/.local/bin/uv run python jobs/my_eval.py --cron"
    }
  }'
```

Key shell preamble details:
- `source /home/fishhouses/.secrets` + `.exports` — injects env vars (Cronicle runs as root, no shell profile)
- XDG overrides — redirects data paths from `/root/` to `/home/fishhouses/`
- `exec uv run` — hands SIGTERM directly to Python

---

## Run Matrix Shape

Each entry in `run_matrix` is a dict with this interface:

```python
{
    "strategy_cls": SomeStrategy,     # instantiated internally by EvalRunner
    "config":       {...},            # passed to strategy; hashed for config_id
    "server":       "deepwater",      # Headwater host alias
    "timeout_s":    300,              # per-doc asyncio.wait_for ceiling
    "concurrency":  3,                # max docs in flight for this entry
    "max_token_count": 100_000,       # optional: skip docs above this token count
}
```

EvalRunner groups entries by server and runs server groups concurrently (different servers run in parallel; entries within one server run serially to avoid overloading Ollama).

---

## DB Storage

All results are persisted via `ConduitDatasetAsync(project_name)`:
- `ds.documents` — source documents (seeded once)
- `ds.runs` — inference outputs (`RunResult`)
- `ds.evals` — judge scores (`EvalResult`)
- `run_failures` table — every failure with error type, token count, traceback

Results are resumable: EvalRunner skips `(strategy, config_id, source_id)` triples already in `ds.runs`.

---

## Key Files

| File | Role |
|---|---|
| `evals/runner.py` | `EvalRunner` base, `ServerCircuitBreaker`, failure persistence |
| `evals/effective_context_window.py` | ECW eval: `ECWEvalRunner`, `BINS`, `assign_bin`, `compute_degradation_curve` |
| `evals/dataset.py` | `ConduitDatasetAsync` — storage layer |
| `evals/scorer.py` | `make_gemini_judge` — Gemini-based scoring |
| `evals/load_datasets.py` | `load_golden_dataset` — 200-doc gold standard corpus |
| `evals/persist.py` | DB connection pool |
| `jobs/effective_context_window.py` | ECW job entry point |
| `jobs/run2.py` | Strategy comparison job entry point |
