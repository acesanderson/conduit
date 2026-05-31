# Evals Architecture

## Three Layers

| Layer | Directory | Purpose |
|---|---|---|
| **Scaffolding** | `evals/` | Reusable infrastructure: `EvalRunner`, `ConduitDatasetAsync`, `make_gemini_judge`, `load_golden_dataset`, `persist`, `nas` |
| **Abstractions** | `evals/` | Reusable eval types: one file per eval methodology, e.g. `effective_context_window.py` |
| **Jobs** | `jobs/` | Runnable entry points: one file per job, thin callers that wire an abstraction to specific models/servers |

Imports go downward only: `jobs/` → `evals/` abstractions → `evals/` scaffolding.

---

## Artifact Storage — $NAS/evals/

All eval outputs (results CSV, status JSON, run log) go to NAS, **not** the local `jobs/` directory. The NAS is mounted at `$NAS` on every host (universal env var; resolves to `/mnt/nas` on Linux, `/Volumes/nas` on macOS).

**Layout**:

```
$NAS/evals/<nas_project>/<eval_name>/
    <eval_name>__<utc_ts>__<host>__results.csv
    <eval_name>__<utc_ts>__<host>__status.json
    <eval_name>__<utc_ts>__<host>__run.log
```

- `nas_project` = repo name (e.g. `conduit`)
- `eval_name` = script-level identifier (e.g. `run2`, `ecw_sweep`)
- `utc_ts` = compact ISO `YYYYMMDDTHHMMSSZ`, set at script start
- `host` = short hostname (alphablue, petrosian, caruana)

Filenames declare provenance: which eval ran, when, on which host. Runs accumulate as history; two hosts running the same eval do not collide.

**Use `evals/nas.py`** in every job:

```python
from nas import artifact_paths
_paths = artifact_paths(nas_project="conduit", eval_name="run2")
RESULTS_PATH = _paths["results_csv"]
STATUS_PATH  = _paths["status_json"]
LOG_PATH     = _paths["log"]
```

**Fail-fast contract**: if `$NAS` is unset or `$NAS/evals/` is missing (NAS unmounted), the helper raises `SystemExit` at import. Jobs do not silently fall back to local storage. Fix the mount, do not paper over.

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

Just `main()` — parse args, build `run_matrix`, instantiate your runner, call `asyncio.run(runner.run(...))`. Wire NAS artifact paths via the helper (never hardcode local paths):

```python
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
from my_eval import MyEvalRunner
from nas import artifact_paths

_paths = artifact_paths(nas_project="conduit", eval_name="my_eval")
RESULTS_PATH = _paths["results_csv"]
STATUS_PATH  = _paths["status_json"]
LOG_PATH     = _paths["log"]
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
