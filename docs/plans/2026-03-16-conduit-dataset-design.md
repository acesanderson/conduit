# ConduitDataset Design

**Date:** 2026-03-16
**Status:** Approved for implementation

---

## 1. Goal

`ConduitDataset` is a project-scoped handle for managing the three stages of an eval pipeline — documents, runs, and evals — backed exclusively by PostgreSQL. It provides async-first access via `ConduitDatasetAsync`, a sync wrapper (`ConduitDatasetSync`) for scripting contexts, and a `ConduitDataset` alias pointing to the sync wrapper. A companion CLI (`conduit-dataset`) provides human-readable inventory snapshots.

---

## 2. Constraints and Non-Goals

**In scope:**
- All reads and writes go to the `evals` PostgreSQL database
- Pool is shared via module-level singleton in `persist.py`; optionally injectable for testing
- The CLI is a registered entrypoint at `src/conduit/apps/scripts/datasets_cli.py`
- Schema migration: `persist.py` DDL is updated to include `project` on `run_results`/`eval_results` and a new `configs` KV table
- `ConduitDatasetSync` implements `__enter__`/`__exit__` so it can be used as a context manager
- A module-level `list_projects()` function (or `ConduitDatasetAsync.list_projects()` classmethod) that returns all distinct project names — required by the CLI `status` command

**Non-goals — explicitly out of scope:**
- Parquet read/write — remains in `load_datasets.py` for intermediate corpus prep
- Versioned/frozen eval splits — `project` tag is the only namespacing mechanism
- Cross-project queries — all list/save operations are scoped to the instantiated project
- Authentication or multi-user access control
- Streaming large result sets — `.list()` methods load fully into memory
- `ConduitDataset` does NOT validate that a `RunResult`'s `source_id` exists in the project's documents — caller's responsibility
- No filtering by date range in this version
- No retry logic on connection failure — surfaces immediately as `ConnectionError`
- No explicit `create_project()` — projects are implicit; a project comes into existence when its first document is saved
- No all-or-nothing batch transactions on `runs.save()` — each row is upserted independently
- No `--json` flag on CLI in this version
- No `list_projects()` method on namespace sub-objects — only on `ConduitDatasetAsync` directly

---

## 3. Interface Contracts

### 3.1 Schema

```sql
CREATE TABLE configs (
    config_id   TEXT        PRIMARY KEY,         -- MD5[:8] hash of config dict
    config      JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    doc_hash    TEXT        NOT NULL,             -- SHA256 of text
    project     TEXT        NOT NULL,
    source_id   TEXT        NOT NULL,
    text        TEXT        NOT NULL,
    reference   TEXT,                             -- gold-standard summary, nullable
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, source_id)
);
CREATE INDEX idx_documents_project  ON documents (project);
CREATE INDEX idx_documents_doc_hash ON documents (doc_hash);

CREATE TABLE run_results (
    project         TEXT        NOT NULL,
    strategy        TEXT        NOT NULL,
    config_id       TEXT        NOT NULL  REFERENCES configs (config_id),
    source_id       TEXT        NOT NULL,
    reference_id    TEXT,
    output          TEXT        NOT NULL,
    output_metadata JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, strategy, config_id, source_id)
);
CREATE INDEX idx_run_results_project   ON run_results (project);
CREATE INDEX idx_run_results_strategy  ON run_results (strategy);
CREATE INDEX idx_run_results_source_id ON run_results (source_id);

CREATE TABLE eval_results (
    project         TEXT        NOT NULL,
    strategy        TEXT        NOT NULL,
    config_id       TEXT        NOT NULL,
    source_id       TEXT        NOT NULL,
    eval_function   TEXT        NOT NULL,
    score           FLOAT       NOT NULL,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, strategy, config_id, source_id, eval_function),
    FOREIGN KEY (project, strategy, config_id, source_id)
        REFERENCES run_results (project, strategy, config_id, source_id)
        ON DELETE CASCADE
);
CREATE INDEX idx_eval_results_project ON eval_results (project);
```

**Schema notes:**
- `run_results` has NO FK to `documents`. Project scoping on runs is enforced by the application layer (`project` column injected from the `ConduitDataset` instance), not by a relational constraint.
- `eval_results` cascades on deletion of its parent `run_results` row. `runs.delete_all()` therefore removes both run and eval rows — calling `evals.delete_all()` first is unnecessary and must NOT be required by the implementation.

### 3.2 File locations

```
evals/
  dataset.py                          ← ConduitDatasetAsync, ConduitDatasetSync,
                                        ConduitDataset alias, namespace classes,
                                        custom exceptions
  persist.py                          ← Updated DDL + pool singleton (raw SQL layer)

src/conduit/apps/scripts/
  datasets_cli.py                     ← CLI entrypoint registered in pyproject.toml
```

### 3.3 Custom exceptions

```python
class DocumentsHaveRunsError(Exception): ...
class GoldStandardExistsError(Exception): ...
class DocumentNotFoundError(Exception): ...
```

All three are defined in `dataset.py` and importable from there.

### 3.4 Class structure

```python
class DocumentsNamespace:
    def __init__(self, project: str, pool_fn: Callable): ...

class RunsNamespace:
    def __init__(self, project: str, pool_fn: Callable): ...

class EvalsNamespace:
    def __init__(self, project: str, pool_fn: Callable): ...

class ConduitDatasetAsync:
    def __init__(self, project: str, pool: asyncpg.Pool | None = None): ...
    documents: DocumentsNamespace
    runs: RunsNamespace
    evals: EvalsNamespace

    @classmethod
    async def list_projects(cls, pool: asyncpg.Pool | None = None) -> list[str]:
        # Returns distinct project names from documents table, sorted alphabetically.
        ...

class ConduitDatasetSync:
    """
    Sync wrapper. Creates a new event loop via asyncio.new_event_loop() at
    __init__ and reuses it for every method call. Raises RuntimeError at
    __init__ if asyncio.get_event_loop().is_running() is True.
    Supports use as a context manager (__enter__/__exit__ call close()).
    """
    def __init__(self, project: str, pool: asyncpg.Pool | None = None): ...
    documents: <sync-wrapped DocumentsNamespace>
    runs: <sync-wrapped RunsNamespace>
    evals: <sync-wrapped EvalsNamespace>
    def close(self) -> None: ...
    def __enter__(self) -> ConduitDatasetSync: ...
    def __exit__(self, *args) -> None: ...   # calls close()

    @classmethod
    def list_projects(cls) -> list[str]: ...  # sync wrapper of async classmethod

ConduitDataset = ConduitDatasetSync
```

### 3.5 Documents namespace — full signatures

```python
# All methods are async on ConduitDatasetAsync; blocking on ConduitDataset/Sync.

async def list() -> list[RunInput]
# Returns all documents for the project ordered by source_id.

async def save(items: list[RunInput]) -> None
# Upsert. On conflict (project, source_id): updates doc_hash, text, metadata.
# Does NOT update reference via this method — reference is preserved on conflict.
# Duplicate source_ids within the batch: last occurrence in the list wins
# (standard executemany behaviour; no error is raised).
# Empty items list: no-op, no error.

async def swap(items: list[RunInput], force: bool = False) -> None
# Atomic: DELETE all documents for project + INSERT items in a single transaction.
# Rolls back entirely if the INSERT raises — original documents are restored.
# Raises DocumentsHaveRunsError if run_results rows exist for this project
# and force=False.
# force=True: deletes run_results for the project first (eval_results cascade
# automatically via FK), then proceeds with the atomic swap.

async def delete_all(force: bool = False) -> None
# Raises DocumentsHaveRunsError if run_results rows exist for the project
# and force=False.
# force=True: deletes run_results first (eval_results cascade), then documents.
# Does NOT call evals.delete_all() separately — relies on FK cascade.

async def needs_gold_standard() -> list[RunInput]
# Returns documents where reference IS NULL, ordered by source_id.

async def save_gold_standards(items: list[RunInput], force: bool = False) -> None
# Validates ALL items before writing ANY (pre-validate, then batch write).
# Validation order:
#   1. Raises DocumentNotFoundError if any source_id does not exist in documents
#      for this project.
#   2. Raises GoldStandardExistsError if any source_id already has a non-null
#      reference and force=False.
# Items where RunInput.reference is None are a no-op for that item (not an error).
# On success: updates ONLY the reference column for each source_id.
# force=True: overwrites existing references without error.
```

### 3.6 Runs namespace — full signatures

```python
async def list(
    strategy: str | None = None,
    config_id: str | None = None,
    source_id: str | None = None,
    config: dict | None = None,
    # config is resolved to config_id via MD5 hash; mutually exclusive with config_id.
    # If the resolved config_id has never been saved, returns [] (not an error).
) -> list[RunResult]
# All filters are ANDed. config and config_id are mutually exclusive (raises ValueError).
# Returns [] for any filter combination that matches no rows.

async def save(results: list[RunResult]) -> None
# For each RunResult:
#   1. Upserts config into configs table (ON CONFLICT DO NOTHING).
#   2. Upserts run_results row.
# Each row is upserted independently — NOT wrapped in a batch transaction.
# Project is injected from the ConduitDataset instance.
# ON CONFLICT (project, strategy, config_id, source_id):
#   updates output, output_metadata, created_at.
# Empty results list: no-op.

async def delete_all() -> None
# Deletes all run_results rows for the project.
# eval_results cascade automatically via FK — do NOT call evals.delete_all() first.

async def list_configs() -> list[dict]
# Returns distinct config dicts used in run_results for this project.
# Ordered by first-seen (configs.created_at ASC).
# Returns [] if no runs exist for the project.
```

### 3.7 Evals namespace — full signatures

```python
async def list(
    eval_function: str | None = None,
    strategy: str | None = None,
) -> list[EvalResult]
# All filters are ANDed. Returns [] for any combination that matches no rows.

async def save(results: list[EvalResult], eval_function: str) -> None
# For each EvalResult, executes in a single database transaction:
#   1. Config upsert (ON CONFLICT DO NOTHING).
#   2. RunResult upsert (same as runs.save() for that row).
#   3. EvalResult upsert.
# All three steps succeed or none are committed (per-result atomicity).
# Failures on one result do not abort the rest of the batch — processing
# continues and all per-result exceptions are collected and re-raised together
# as a single BatchSaveError at the end.
# ON CONFLICT (project, strategy, config_id, source_id, eval_function):
#   updates score, metadata, created_at.
# Empty results list: no-op.

async def delete_all() -> None
# Deletes all eval_results rows for the project.
# Does NOT touch run_results.
```

### 3.8 BatchSaveError

```python
class BatchSaveError(Exception):
    """Raised by evals.save() when one or more per-result transactions fail."""
    failures: list[tuple[EvalResult, Exception]]
    # Each entry is the EvalResult that failed and the exception that caused it.
```

### 3.9 CLI

```
conduit-dataset status
# Lists all distinct projects with counts for each stage (documents, runs, evals).
# Exits 1 with "Cannot reach postgres at <host>:<port>" if DB unreachable.
# Output to stdout.

conduit-dataset inspect <project>
# Detailed breakdown for one project: counts by stage.
# Exits 1 if project does not exist.
# Exits 1 with DB error message if unreachable.

conduit-dataset inspect <project> --scores
# Same as inspect, plus mean/min/max score per (strategy, eval_function).
```

---

## 4. Acceptance Criteria

All criteria assume the `evals` database is reachable and tables exist.

1. Given 3 documents saved for project `"test"` (two with `reference`, one without), `cd.documents.list()` returns exactly 3 `RunInput`s with `reference` populated on the two that have it and `None` on the third.
2. `documents.swap(items)` is atomic: given a failing INSERT (e.g. constraint violation), querying documents afterwards returns the original pre-swap rows unchanged.
3. `documents.delete_all()` with existing `run_results` rows raises `DocumentsHaveRunsError`. `delete_all(force=True)` leaves zero documents, zero runs, and zero evals for the project.
4. `documents.save(items)` does NOT overwrite an existing non-null `reference` on conflict — the reference value is preserved after a second save of the same document with `reference=None`.
5. `documents.save_gold_standards(items)` raises `GoldStandardExistsError` when any item already has a reference and `force=False`. No rows are written when the error is raised (pre-validate behaviour). `force=True` overwrites successfully.
6. `documents.save_gold_standards(items)` raises `DocumentNotFoundError` when any `source_id` is not present in the project. No rows are written when the error is raised.
7. `documents.save_gold_standards(items)` where some `RunInput.reference` is `None` — those items are skipped silently; other items with non-null references are written.
8. `runs.list(config={"model": "gpt-oss:latest"})` returns only runs whose `config_id` equals `hashlib.md5(json.dumps({"model": "gpt-oss:latest"}, sort_keys=True).encode()).hexdigest()[:8]`.
9. `runs.list(config={"model": "unknown"})` returns `[]` (no error) when that config has never been saved.
10. `runs.list(config=..., config_id=...)` raises `ValueError` at call site.
11. `runs.list_configs()` returns configs ordered by `configs.created_at ASC` and contains no duplicates.
12. `evals.save(results, eval_function)` — given one result whose RunResult has an invalid `config_id` (FK violation), that result's transaction is rolled back, the rest of the batch succeeds, and a `BatchSaveError` is raised containing exactly that one failure.
13. `configs` table has exactly one row per unique config dict regardless of how many runs share that config.
14. `ConduitDataset("x")` raises `RuntimeError("Use ConduitDatasetAsync in async contexts")` when instantiated inside a running event loop.
15. `ConduitDataset` method call after `close()` raises `RuntimeError("ConduitDataset is closed")`.
16. `conduit-dataset status` exits with code 1 and prints a message containing the host and database name when postgres is unreachable.
17. All upserts (documents, runs, evals) are idempotent: saving the same data twice produces exactly one row with the latest `created_at`.
18. `ConduitDataset` used as a context manager (`with ConduitDataset("x") as cd:`) calls `close()` on exit even if the body raises.

---

## 5. Error Handling / Failure Modes

| Situation | Behaviour |
|---|---|
| Postgres unreachable | `ConnectionError` with host + port + dbname in message. CLI exits 1 with same message to stdout. |
| Pool exhausted | asyncpg raises; surfaces as-is to caller. Not retried. |
| `delete_all(force=False)` with existing runs | `DocumentsHaveRunsError` |
| `save_gold_standards()` — any source_id missing from project | `DocumentNotFoundError`; no rows written |
| `save_gold_standards()` — any reference already exists, `force=False` | `GoldStandardExistsError`; no rows written |
| `save_gold_standards()` — `RunInput.reference` is `None` | Silent no-op for that item |
| `runs.list(config=..., config_id=...)` | `ValueError` at call site, before any DB query |
| `runs.list(config=unknown_dict)` | `[]` returned, no error |
| `ConduitDataset()` called inside running event loop | `RuntimeError("Use ConduitDatasetAsync in async contexts")` |
| `ConduitDatasetSync` method call after `close()` | `RuntimeError("ConduitDataset is closed")` |
| `evals.save()` — per-result transaction fails | That result rolled back; rest of batch continues; `BatchSaveError` raised at end with all failures |
| `swap()` — DELETE succeeds, INSERT fails | Transaction rolled back; original documents restored |
| Empty `items` or `results` list | No-op, no error |
| `documents.save()` with duplicate `source_id`s in batch | Last occurrence wins; no error raised |

---

## 6. Observability

### Logging
All logging uses the standard `logging` module. Logger name: `conduit.dataset`.

| Event | Level | Required fields |
|---|---|---|
| Pool acquired (first connection) | `DEBUG` | host, dbname |
| `save()` called | `DEBUG` | project, operation, item count |
| `delete_all(force=True)` called | `WARNING` | project, operation |
| `swap()` called | `INFO` | project, item count |
| Per-result transaction failure in `evals.save()` | `ERROR` | project, strategy, config_id, source_id, exception message |
| `BatchSaveError` raised | `ERROR` | project, total failures, total attempted |
| `GoldStandardExistsError` raised | `WARNING` | project, source_ids that already had references |

No `print()` statements anywhere in `dataset.py` or `persist.py`.

### CLI exit codes
| Situation | Exit code |
|---|---|
| Success | 0 |
| DB unreachable | 1 |
| Project not found (`inspect`) | 1 |
| Unexpected error | 2 |

---

## 7. Code Style Example

```python
# Async (in eval scripts)
async def run():
    cd = ConduitDatasetAsync("summarize")
    docs = await cd.documents.needs_gold_standard()
    enriched = await generate_gold_standards(docs)
    await cd.documents.save_gold_standards(enriched)

# Sync (in corpus prep scripts) — as context manager
def prepare():
    with ConduitDataset("summarize") as cd:    # alias for ConduitDatasetSync
        items = load_from_parquet(CORPUS_PATH) # → list[RunInput]
        cd.documents.save(items)

# List all projects (for CLI status)
projects = await ConduitDatasetAsync.list_projects()
```

---

## 8. Domain Language

| Term | Meaning |
|---|---|
| **project** | A named namespace (`str`) grouping documents, runs, and evals. Created implicitly when the first document is saved. E.g. `"summarize"`. |
| **document** | A `RunInput` stored in the `documents` table. Has `source_id`, `text` (`data`), optional `reference`, optional `metadata`. |
| **gold standard** | The `reference` field on a document: a human- or model-generated authoritative summary. |
| **run** | A `RunResult` stored in `run_results`. One execution of one strategy against one document under one config. |
| **eval** | An `EvalResult` stored in `eval_results`. One score assigned to one run by one eval function. |
| **config** | A `dict` stored in the `configs` KV table, keyed by `config_id`. |
| **config_id** | The first 8 hex chars of `hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()`. |
| **doc_hash** | `hashlib.sha256(text.encode()).hexdigest()`. Used for deduplication, not as a PK. |
| **namespace** | `DocumentsNamespace`, `RunsNamespace`, `EvalsNamespace` — injected sub-objects on `ConduitDatasetAsync`. |
| **pool** | An `asyncpg.Pool`. Module-level singleton by default; injectable for tests. |
| **batch** | The full `list[RunResult]` or `list[EvalResult]` passed to a single `save()` call. |

---

## 9. Invalid State Transitions

These must raise and never silently succeed:

1. **`documents.delete_all(force=False)`** when `run_results` rows exist for the project.
2. **`documents.swap(items, force=False)`** when `run_results` rows exist for the project.
3. **`documents.save_gold_standards(items, force=False)`** when any item's `source_id` already has a non-null `reference`.
4. **`documents.save_gold_standards(items)`** when any item's `source_id` does not exist in `documents` for this project.
5. **`runs.list(config=..., config_id=...)`** — both filter arguments supplied simultaneously.
6. **`ConduitDatasetSync.__init__`** when `asyncio.get_event_loop().is_running()` is `True`.
7. **`ConduitDatasetSync` method call after `close()`** — event loop is closed; must raise `RuntimeError("ConduitDataset is closed")`.
