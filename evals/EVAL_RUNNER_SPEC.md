# EvalRunner — Design Spec

## Motivation

`run2.py` was written as a one-off strategy comparison script, but over several iterations it accumulated a substantial layer of orchestration infrastructure: health checks, circuit breakers, per-doc checkpointing, failure persistence, enriched logging, and status reporting. That infrastructure is task-agnostic. The task-specific parts — the run matrix, the dataset loader, the judge — are a thin layer on top.

This spec describes extracting the infrastructure into a reusable `EvalRunner` base so that future eval scripts are 50 lines, not 500, and the diagnostic machinery is ambient rather than something each script has to carry.

---

## Abstraction Boundary

### Generic (moves into `EvalRunner`)

| Component | Notes |
|---|---|
| Health check + warmup gate | Parameterized from server names in the run matrix |
| `ServerCircuitBreaker` | Per-server; derived automatically from matrix |
| `run_failures` table + `_save_failure` + `_classify_error` | Zero task coupling |
| `_run_inference_incremental` | Core loop: semaphore, per-doc saves, failure recording |
| `score_missing` | Operates on `run_results`/`eval_results` tables generically |
| `run_entry` | Orchestrates one (callable, config) cell with resumability + smoke gate |
| Status file (`*_status.json`) | Success/failure record for Cronicle |
| macOS notification | Completion signal |
| `--cron`, `--dry-run`, `--limit`, `--project` CLI flags | Standard across all evals |

### Task-specific (provided by caller)

| Component | Notes |
|---|---|
| `run_matrix` | List of (strategy, config, server, timeout_s, concurrency) cells |
| `dataset_loader` | `() -> list[RunInput]` |
| `judge_factory` | `(references: dict) -> EvalFunction` |
| `project` | DB project name string |
| `publish()` | Optional hook; results analysis and export (e.g. xlsx). Not worth abstracting — callers override or omit. |

---

## Interface

```python
runner = EvalRunner(
    run_matrix=RUN_MATRIX,
    dataset_loader=load_golden_dataset,
    judge_factory=make_gemini_judge,
    project="run2_strategy_comparison",
)
asyncio.run(runner.run())
```

`EvalRunner.__init__` derives server names from the matrix and initialises one `ServerCircuitBreaker` per unique server. Callers never touch circuit breakers directly.

`runner.run()` is the full pipeline: health check → seed documents → inference (resumable) → score missing → write status → notify.

`runner.publish()` is a no-op by default. Subclasses or callers override it to add analysis, CSV/xlsx export, or anything task-specific. It is called at the end of a successful run.

---

## Run Matrix Shape

Each entry is a dict with a required interface:

```python
{
    "strategy_cls": SomeStrategy,   # instantiated internally
    "config":       {...},          # passed to strategy; hashed for config_id
    "server":       "deepwater",    # used for circuit breaker + health check routing
    "timeout_s":    1800,           # per-doc asyncio.wait_for ceiling
    "concurrency":  1,              # max docs in flight simultaneously for this entry
}
```

The resumability key is derived as `(strategy.__class__.__name__, config_id, source_id)` — same as now. If a future eval has a shape that doesn't fit "strategy class + config dict," the key should be made explicit in the matrix entry rather than derived.

---

## Cronicle Compatibility

No changes to the existing contract:

- `--cron` flag: ping + warmup servers; `sys.exit(0)` if unreachable (don't count as job failure)
- Exit code 0 = success or graceful skip; non-zero = unexpected failure
- `{project}_status.json` written on completion (success or failure) for post-run inspection
- Progress printed to stdout; Cronicle captures it as job output
- Log file written alongside the script (e.g. `run2.log`) via `logging.FileHandler`

---

## Open Design Questions

**1. Is `ConduitDatasetAsync` always the storage layer?**

Currently `run_entry` is tightly coupled to `ConduitDatasetAsync`. For conduit-based evals this is fine — make it a required dependency of `EvalRunner`. If evals against raw API calls or non-conduit models are needed later, replace with a storage protocol. Not worth solving now.

**2. How general is the matrix shape?**

The current shape assumes "strategy × config." Other eval types that fit the same infrastructure but not the same shape:

- *Prompt A/B test*: prompt A vs. prompt B, same model. Fits if "strategy" is interpreted as "a callable that applies a prompt."
- *Dataset quality audit*: one strategy, fixed config, pass/fail per doc. Fits trivially (matrix with one entry).
- *Regression test*: run the same strategy before/after a code change. Fits if project names are versioned.

The matrix-of-dicts approach handles all of these without schema changes. The looser alternative — "list of named callables" — is more general but loses automatic key derivation for resumability.

**3. Smoke gate configurability**

Currently hardcoded to 2 docs. Should be a constructor parameter with 2 as default.

---

## Migration Path

1. Extract infrastructure from `run2.py` into `evals/runner.py` (~300 lines)
2. Rewrite `run2.py` as a thin caller (~50 lines): define matrix, loader, judge, instantiate `EvalRunner`, run
3. Future eval scripts follow the same pattern from the start
4. `publish_results.py` becomes either the default `publish()` implementation or a standalone script callers invoke after the run
