# EvalRunner Extraction + Effective Context Window Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract run2.py's orchestration infrastructure into a reusable `EvalRunner` base class, reorganize eval scripts under `jobs/`, and add a new `effective_context_window.py` eval that measures summarization quality degradation across token-length bins to identify a model's effective context window.

**Architecture:** `EvalRunner` base lives in `evals/runner.py` and handles all orchestration (circuit breakers, failure persistence, health checks, per-doc checkpointing, status files). Job scripts in `jobs/` are thin callers (~50 lines): define a run matrix, instantiate a subclass with a `publish()` override, call `asyncio.run(runner.run(...))`. The ECW eval takes `--model` and `--server` as required CLI flags; one Cronicle event is created per model being tested.

**Tech Stack:** Python 3.11+, asyncio, asyncpg (via `ConduitDatasetAsync`), pandas, statistics stdlib, headwater_client, conduit summarizers, argparse.

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Create | `evals/runner.py` | `EvalRunner` base + all infrastructure (`ServerCircuitBreaker`, failure persistence, inference loop, health check) |
| Create | `evals/tests/test_runner.py` | Unit tests for `ServerCircuitBreaker`, `_classify_error`, `_config_id` |
| Create | `evals/tests/test_effective_context_window.py` | Unit tests for `assign_bin`, `compute_degradation_curve` |
| Create | `jobs/run2.py` | Thin run2 caller; `publish()` override contains current `print_results` logic |
| Create | `jobs/effective_context_window.py` | ECW eval entry point; `publish()` produces score-by-bin degradation curve |
| Delete | `evals/run2.py` | Replaced by `jobs/run2.py` |

---

### Task 1: Unit tests for EvalRunner infrastructure

**Files:**
- Create: `evals/tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# evals/tests/test_runner.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import ServerCircuitBreaker, _classify_error, _config_id


def test_config_id_is_deterministic():
    cfg = {"model": "qwen3.6:latest", "use_remote": True, "host_alias": "deepwater"}
    assert _config_id(cfg) == _config_id(cfg)


def test_config_id_differs_for_different_configs():
    assert _config_id({"model": "qwen3.6:latest"}) != _config_id({"model": "gpt-oss:latest"})


def test_classify_timeout():
    assert _classify_error(asyncio.TimeoutError()) == "timeout"


def test_classify_network_error():
    assert _classify_error(Exception("NETWORK_ERROR: connection refused")) == "network_error"


def test_classify_context_overflow():
    assert _classify_error(Exception("INTERNAL_ERROR: empty response from model")) == "context_overflow"


def test_classify_context_overflow_num_ctx():
    assert _classify_error(Exception("INTERNAL_ERROR: num_ctx exceeded")) == "context_overflow"


def test_classify_internal_error():
    assert _classify_error(Exception("INTERNAL_ERROR: something else")) == "internal_error"


def test_classify_connection_error():
    assert _classify_error(ConnectionError("Cannot connect to server")) == "connection_error"


def test_classify_unknown():
    assert _classify_error(ValueError("some random error")) == "ValueError"


def test_circuit_breaker_opens_after_threshold():
    cb = ServerCircuitBreaker("testserver", threshold=3, cooldown_s=60.0)

    async def run():
        for _ in range(3):
            await cb.record_failure()
        assert cb._opened_at is not None

    asyncio.run(run())


def test_circuit_breaker_closes_on_success():
    cb = ServerCircuitBreaker("testserver", threshold=3, cooldown_s=60.0)

    async def run():
        for _ in range(3):
            await cb.record_failure()
        assert cb._opened_at is not None
        await cb.record_success()
        assert cb._opened_at is None

    asyncio.run(run())


def test_circuit_breaker_does_not_open_below_threshold():
    cb = ServerCircuitBreaker("testserver", threshold=3, cooldown_s=60.0)

    async def run():
        for _ in range(2):
            await cb.record_failure()
        assert cb._opened_at is None

    asyncio.run(run())


def test_circuit_breaker_resets_consecutive_on_success():
    cb = ServerCircuitBreaker("testserver", threshold=5, cooldown_s=60.0)

    async def run():
        for _ in range(2):
            await cb.record_failure()
        await cb.record_success()
        assert cb._consecutive == 0

    asyncio.run(run())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python -m pytest evals/tests/test_runner.py -v
```

Expected: `ImportError` — `runner` module not yet created.

---

### Task 2: Create `evals/runner.py`

**Files:**
- Create: `evals/runner.py`

- [ ] **Step 1: Write EvalRunner**

```python
# evals/runner.py
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import statistics
import subprocess
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataset import ConduitDatasetAsync
    from evals import EvalResult, RunInput, RunResult

logger = logging.getLogger(__name__)

EVAL_FUNCTION = "gemini_judge"


def _config_id(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def _classify_error(exc: BaseException) -> str:
    msg = str(exc)
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if "NETWORK_ERROR" in msg:
        return "network_error"
    if "INTERNAL_ERROR" in msg:
        if "empty response" in msg.lower() or "num_ctx" in msg.lower():
            return "context_overflow"
        return "internal_error"
    if isinstance(exc, ConnectionError) or "Cannot connect" in msg:
        return "connection_error"
    return type(exc).__name__


_FAILURES_DDL = """
CREATE TABLE IF NOT EXISTS run_failures (
    project       TEXT        NOT NULL,
    strategy      TEXT        NOT NULL,
    config_id     TEXT        NOT NULL,
    source_id     TEXT        NOT NULL,
    error_type    TEXT        NOT NULL,
    error_message TEXT,
    token_count   INTEGER,
    traceback     TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_run_failures_project ON run_failures (project);
CREATE INDEX IF NOT EXISTS idx_run_failures_error   ON run_failures (error_type);
CREATE INDEX IF NOT EXISTS idx_run_failures_source  ON run_failures (source_id);
"""


class ServerCircuitBreaker:
    """
    Opens after `threshold` consecutive failures; blocks new dispatches until
    `cooldown_s` has elapsed.
    """

    def __init__(self, server: str, threshold: int = 5, cooldown_s: float = 60.0):
        self.server = server
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._consecutive = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            self._consecutive = 0
            if self._opened_at is not None:
                logger.info("circuit %s: CLOSED on success", self.server)
                self._opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._consecutive += 1
            if self._consecutive >= self.threshold and self._opened_at is None:
                self._opened_at = asyncio.get_event_loop().time()
                logger.warning(
                    "circuit %s: OPEN after %d consecutive failures — %.0fs cooldown",
                    self.server, self._consecutive, self.cooldown_s,
                )

    async def wait_if_open(self) -> None:
        while True:
            async with self._lock:
                if self._opened_at is None:
                    return
                elapsed = asyncio.get_event_loop().time() - self._opened_at
                if elapsed >= self.cooldown_s:
                    logger.info("circuit %s: RESET after %.0fs cooldown", self.server, elapsed)
                    self._opened_at = None
                    self._consecutive = 0
                    return
                remaining = self.cooldown_s - elapsed
            logger.info("circuit %s: open, waiting %.0fs", self.server, remaining)
            await asyncio.sleep(min(remaining, 5.0))


class EvalRunner:
    def __init__(
        self,
        run_matrix: list[dict],
        dataset_loader: Callable[[], list[RunInput]],
        judge_factory: Callable[[dict], Any],
        project: str,
        log_path: Path,
        status_path: Path,
        smoke_gate: int = 2,
    ) -> None:
        self._run_matrix = run_matrix
        self._dataset_loader = dataset_loader
        self._judge_factory = judge_factory
        self._project = project
        self._log_path = log_path
        self._status_path = status_path
        self._smoke_gate = smoke_gate
        # Derive unique servers + a representative model per server for warmup
        self._servers: dict[str, str] = {}
        for entry in run_matrix:
            alias = entry["server"]
            if alias not in self._servers:
                self._servers[alias] = entry["config"]["model"]
        self._circuit_breakers: dict[str, ServerCircuitBreaker] = {
            alias: ServerCircuitBreaker(alias) for alias in self._servers
        }

    def _setup_logging(self) -> None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self._log_path),
            ],
        )

    def _print_matrix(self, docs_count: int | None = None) -> None:
        print(f"Project:   {self._project}")
        if docs_count is not None:
            print(f"Docs:      {docs_count}")
        print(f"Matrix ({len(self._run_matrix)} entries):")
        for e in self._run_matrix:
            cap = f"  cap={e['max_token_count'] // 1000}K" if e.get("max_token_count") else ""
            model = e["config"].get("model", "")
            print(f"  {e['strategy_cls'].__name__:<30} {model:<20} [{e['server']}]{cap}")

    async def _warmup_server(self, alias: str) -> bool:
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        from headwater_api.classes import BatchRequest
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.utils.progress.verbosity import Verbosity

        model = self._servers[alias]
        try:
            params = GenerationParams(model=model, temperature=0.0)
            options = ConduitOptions(
                project_name="warmup",
                include_history=False,
                verbosity=Verbosity.SILENT,
            )
            batch_req = BatchRequest(
                prompt_strings_list=["Hi"],
                params=params,
                options=options,
            )
            async with HeadwaterAsyncClient(host_alias=alias) as client:
                resp = await asyncio.wait_for(
                    client.conduit.query_batch(batch_req), timeout=30.0
                )
            return bool(resp and resp.results)
        except Exception as exc:
            print(f"[cron] {alias} warmup failed: {exc}")
            return False

    async def _health_check(self) -> bool:
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        for alias in self._servers:
            try:
                async with HeadwaterAsyncClient(host_alias=alias) as client:
                    ok = await client.ping()
                if not ok:
                    print(f"[cron] {alias} ping returned False — aborting")
                    return False
            except Exception as exc:
                print(f"[cron] {alias} unreachable: {exc} — aborting")
                return False
            if not await self._warmup_server(alias):
                return False
        return True

    async def _ensure_failures_table(self) -> None:
        from persist import _get_pool
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_FAILURES_DDL)

    async def _save_failure(
        self,
        strategy: str,
        config_id: str,
        source_id: str,
        error_type: str,
        error_message: str,
        token_count: int | None,
        tb_str: str,
    ) -> None:
        from persist import _get_pool
        pool = await _get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO run_failures
                        (project, strategy, config_id, source_id, error_type,
                         error_message, token_count, traceback)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    self._project, strategy, config_id, source_id,
                    error_type, error_message[:500], token_count, tb_str[:4000],
                )
        except Exception as db_exc:
            logger.warning("failed to persist failure record: %s", db_exc)

    def _write_status(self, status: dict) -> None:
        self._status_path.write_text(json.dumps(status, indent=2, default=str))

    def _notify(self, title: str, message: str) -> None:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True)

    async def _run_inference_incremental(
        self,
        docs: list[RunInput],
        config: dict,
        strategy: Any,
        ds: ConduitDatasetAsync,
        timeout_s: int,
        circuit_breaker: ServerCircuitBreaker,
        concurrency: int,
    ) -> list[RunResult]:
        from evals import run_eval
        sem = asyncio.Semaphore(concurrency)
        strategy_name = strategy.__class__.__name__
        cid = _config_id(config)
        inflight: list[int] = [0]

        async def run_and_save(doc: RunInput) -> RunResult | None:
            await circuit_breaker.wait_if_open()
            async with sem:
                inflight[0] += 1
                token_count: int | None = (doc.metadata or {}).get("token_count")
                try:
                    result = await asyncio.wait_for(
                        run_eval(doc, config, strategy), timeout=timeout_s
                    )
                    await ds.runs.save([result])
                    await circuit_breaker.record_success()
                    return result
                except asyncio.TimeoutError as exc:
                    tb_str = traceback.format_exc()
                    logger.warning(
                        "timeout source_id=%s tokens=%s inflight=%d after %ds\n%s",
                        doc.source_id, token_count, inflight[0], timeout_s, tb_str,
                    )
                    await circuit_breaker.record_failure()
                    await self._save_failure(strategy_name, cid, doc.source_id,
                                             "timeout", str(exc), token_count, tb_str)
                    return None
                except Exception as exc:
                    error_type = _classify_error(exc)
                    tb_str = traceback.format_exc()
                    logger.error(
                        "run_eval failed source_id=%s tokens=%s inflight=%d "
                        "strategy=%s error=%s: %s\n%s",
                        doc.source_id, token_count, inflight[0],
                        strategy_name, error_type, exc, tb_str,
                    )
                    await circuit_breaker.record_failure()
                    await self._save_failure(strategy_name, cid, doc.source_id,
                                             error_type, str(exc), token_count, tb_str)
                    return None
                finally:
                    inflight[0] -= 1

        tasks = [asyncio.create_task(run_and_save(doc)) for doc in docs]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    async def _score_missing(
        self,
        ds: ConduitDatasetAsync,
        strategy_name: str,
        cid: str,
        judge: Any,
    ) -> list[EvalResult]:
        from evals import CONCURRENCY_LIMIT, EvalResult
        all_runs = await ds.runs.list(strategy=strategy_name, config_id=cid)
        existing_evals = await ds.evals.list(strategy=strategy_name, config_id=cid)
        scored_ids = {er.run_result.source_id for er in existing_evals}
        unscored = [r for r in all_runs if r.source_id not in scored_ids]

        if not unscored:
            return []

        print(f"  Scoring {len(unscored)} unscored results ...")
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def score_and_save(run_result: RunResult) -> EvalResult | None:
            async with sem:
                try:
                    score = await judge(run_result)
                    er = EvalResult(run_result=run_result, score=score)
                    await ds.evals.save([er], eval_function=EVAL_FUNCTION)
                    return er
                except Exception as exc:
                    logger.error(
                        "scoring failed source_id=%s: %s: %s",
                        run_result.source_id, type(exc).__name__, exc,
                    )
                    return None

        tasks = [asyncio.create_task(score_and_save(r)) for r in unscored]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    async def _run_entry(
        self,
        ds: ConduitDatasetAsync,
        entry: dict,
        docs: list[RunInput],
        judge: Any,
        smoke_tested: set[tuple[str, str]],
    ) -> list[RunResult]:
        from evals import generate_runs
        strategy = entry["strategy_cls"]()
        config = entry["config"]
        strategy_name = strategy.__class__.__name__
        server = entry["server"]
        cid = _config_id(config)
        max_tc = entry.get("max_token_count")
        if max_tc is not None:
            docs = [d for d in docs if (d.metadata or {}).get("token_count", 0) <= max_tc]
        n_total = len(docs)

        done_ids = {r.source_id for r in await ds.runs.list(strategy=strategy_name, config_id=cid)}
        if len(done_ids) >= n_total:
            await self._score_missing(ds, strategy_name, cid, judge)
            return []

        remaining = [d for d in docs if d.source_id not in done_ids]
        if done_ids:
            print(f"  RESUME {strategy_name}/{cid}: {len(done_ids)}/{n_total} done, {len(remaining)} remaining")
        else:
            print(f"  START  {strategy_name}/{cid} x {n_total} docs  [{server}]")

        smoke_key = (strategy_name, server)
        if not done_ids and smoke_key not in smoke_tested:
            print(f"  Smoke: {strategy_name} on {server} x {self._smoke_gate} docs ...")
            try:
                smoke_results = await generate_runs(
                    inputs=remaining[:self._smoke_gate],
                    configs=[config],
                    strategy=strategy,
                )
                if len(smoke_results) < self._smoke_gate:
                    print(f"  ABORT  {strategy_name}/{server} — smoke test failed "
                          f"({len(smoke_results)}/{self._smoke_gate} succeeded)")
                    return []
                print("  Smoke passed.")
                smoke_tested.add(smoke_key)
            except Exception as exc:
                print(f"  ABORT  {strategy_name}/{server} — smoke test error: {exc}")
                logger.exception("smoke test error strategy=%s server=%s", strategy_name, server)
                return []

        run_results = await self._run_inference_incremental(
            docs=remaining,
            config=config,
            strategy=strategy,
            ds=ds,
            timeout_s=entry["timeout_s"],
            circuit_breaker=self._circuit_breakers[server],
            concurrency=entry.get("concurrency", 1),
        )
        print(f"  Done.  {strategy_name}/{cid}: {len(run_results)}/{len(remaining)} succeeded.")
        await self._score_missing(ds, strategy_name, cid, judge)
        return run_results

    async def publish(self, ds: ConduitDatasetAsync, doc_meta: dict[str, dict]) -> None:
        pass

    async def run(
        self,
        *,
        limit: int | None = None,
        cron: bool = False,
        dry_run: bool = False,
    ) -> None:
        from dataset import ConduitDatasetAsync

        self._setup_logging()

        if dry_run:
            self._print_matrix()
            return

        if cron and not await self._health_check():
            sys.exit(0)

        docs = self._dataset_loader()
        if limit:
            docs = docs[:limit]

        self._print_matrix(docs_count=len(docs))

        ds = ConduitDatasetAsync(self._project)
        await ds.documents.save(docs)
        await self._ensure_failures_table()

        references = {doc.source_id: doc.reference for doc in docs}
        judge = self._judge_factory(references)
        doc_meta = {doc.source_id: doc.metadata for doc in docs}

        smoke_tested: set[tuple[str, str]] = set()
        all_results: list[RunResult] = []
        started_at = datetime.now()

        # Group entries by server; run server groups concurrently, entries within
        # each group sequentially (avoids overloading a single Ollama instance).
        by_server: dict[str, list[dict]] = {}
        for entry in self._run_matrix:
            by_server.setdefault(entry["server"], []).append(entry)

        async def run_server_entries(entries: list[dict]) -> list[RunResult]:
            results = []
            for entry in entries:
                batch = await self._run_entry(ds, entry, docs, judge, smoke_tested)
                results.extend(batch)
            return results

        try:
            server_tasks = [
                asyncio.create_task(run_server_entries(entries))
                for entries in by_server.values()
            ]
            server_results = await asyncio.gather(*server_tasks)
            for batch in server_results:
                all_results.extend(batch)

            print(f"\nTotal new results this run: {len(all_results)}")
            await self.publish(ds, doc_meta)

            self._write_status({
                "result": "ok",
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "new_results": len(all_results),
            })
            self._notify(f"{self._project} complete", f"{len(all_results)} new results")

        except Exception as exc:
            self._write_status({
                "result": "failed",
                "started_at": started_at.isoformat(),
                "failed_at": datetime.now().isoformat(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            self._notify(f"{self._project} FAILED", str(exc)[:100])
            raise
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python -m pytest evals/tests/test_runner.py -v
```

Expected: all 12 tests pass.

- [ ] **Step 3: Commit**

```bash
git add evals/runner.py evals/tests/test_runner.py
git commit -m "feat(evals): add EvalRunner base class extracted from run2.py"
```

---

### Task 3: Create `jobs/run2.py` and delete `evals/run2.py`

**Files:**
- Create: `jobs/run2.py`
- Delete: `evals/run2.py`

- [ ] **Step 1: Write `jobs/run2.py`**

```python
# jobs/run2.py
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from dataset import ConduitDatasetAsync
from load_datasets import load_golden_dataset
from scorer import make_gemini_judge
from runner import EVAL_FUNCTION, EvalRunner
from conduit.strategies.summarize.summarizers.hierarchical_tree import HierarchicalTreeSummarizer
from conduit.strategies.summarize.summarizers.map_dedupe_reduce import MapDedupeReduceSummarizer
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer
from conduit.strategies.summarize.summarizers.rolling_refine import RollingRefineSummarizer

PROJECT = "run2_strategy_comparison"
RESULTS_PATH = Path(__file__).parent / "run2_results.csv"
LOG_PATH = Path(__file__).parent / "run2.log"
STATUS_PATH = Path(__file__).parent / "run2_status.json"

_QWEN = {"model": "qwen3.6:latest", "use_remote": True, "host_alias": "deepwater", "use_cache": True}
_QWEN_RECURSIVE = {**_QWEN, "map_model": "gpt-oss:latest", "map_host_alias": "bywater"}
_GPT = {"model": "gpt-oss:latest", "use_remote": True, "host_alias": "bywater", "use_cache": True}
_GEMMA = {"model": "gemma4:latest", "use_remote": True, "host_alias": "deepwater", "use_cache": True}
_GEMMA_RECURSIVE = {**_GEMMA, "map_model": "gpt-oss:latest", "map_host_alias": "bywater"}

RUN_MATRIX = [
    {"strategy_cls": RecursiveSummarizer,        "config": _QWEN_RECURSIVE,  "server": "deepwater", "timeout_s": 600,  "concurrency": 5},
    {"strategy_cls": RollingRefineSummarizer,    "config": _QWEN,            "server": "deepwater", "timeout_s": 3600, "concurrency": 1},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _QWEN,            "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _QWEN,            "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    {"strategy_cls": RecursiveSummarizer,        "config": _GPT,             "server": "bywater",   "timeout_s": 600,  "concurrency": 5},
    {"strategy_cls": RollingRefineSummarizer,    "config": _GPT,             "server": "bywater",   "timeout_s": 2400, "concurrency": 1},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _GPT,             "server": "bywater",   "timeout_s": 1200, "concurrency": 2},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _GPT,             "server": "bywater",   "timeout_s": 1200, "concurrency": 2},
    {"strategy_cls": RecursiveSummarizer,        "config": _GEMMA_RECURSIVE, "server": "deepwater", "timeout_s": 600,  "concurrency": 5},
    {"strategy_cls": RollingRefineSummarizer,    "config": _GEMMA,           "server": "deepwater", "timeout_s": 3600, "concurrency": 1},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _GEMMA,           "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _GEMMA,           "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    {"strategy_cls": OneShotSummarizer, "config": _GEMMA, "server": "deepwater", "timeout_s": 300, "concurrency": 3, "max_token_count": 100_000},
    {"strategy_cls": OneShotSummarizer, "config": _GPT,   "server": "bywater",   "timeout_s": 300, "concurrency": 5, "max_token_count": 100_000},
]


class Run2EvalRunner(EvalRunner):
    async def publish(self, ds: ConduitDatasetAsync, doc_meta: dict) -> None:
        eval_results = await ds.evals.list(eval_function=EVAL_FUNCTION)
        if not eval_results:
            print("No eval results in DB yet.")
            return

        rows = []
        for er in eval_results:
            r = er.run_result
            meta = doc_meta.get(r.source_id, {})
            config = r.config if isinstance(r.config, dict) else r.config.model_dump()
            trace = r.output.metadata.get("trace", [])
            duration = trace[-1]["duration"] if trace else None
            rows.append({
                "strategy": r.strategy,
                "model": config.get("model", ""),
                "config_id": r.config_id,
                "source_id": r.source_id,
                "category": meta.get("category", ""),
                "token_count": meta.get("token_count", 0),
                "score": er.score,
                "output_chars": len(r.output.output),
                "duration_s": duration,
            })

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_PATH, index=False)
        print(f"\nResults saved to {RESULTS_PATH}")

        CHUNK_SIZE = 12_000
        df_multi = df[df["token_count"] > CHUNK_SIZE].copy()
        n_one_shot = len(df) - len(df_multi)
        n_strategies = df["strategy"].nunique()
        print(f"\nAnalysis: excluding {n_one_shot // n_strategies} one-shot docs "
              f"(token_count <= {CHUNK_SIZE}), using {len(df_multi) // n_strategies} multi-chunk docs")

        summary = (
            df_multi.groupby(["strategy", "model"])["score"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"count": "n"})
            .sort_values("mean", ascending=False)
        )
        print("\n=== Scores by strategy x model (multi-chunk docs only) ===")
        print(summary.round(3).to_string())

        by_category = df_multi.groupby(["strategy", "category"])["score"].mean().unstack("category")
        print("\n=== Scores by strategy x category (multi-chunk docs only) ===")
        print(by_category.round(3).to_string())

        speed = (
            df_multi.dropna(subset=["duration_s"])
            .groupby(["strategy", "model"])["duration_s"]
            .agg(["mean", "median", "max"])
            .sort_values("mean")
        )
        print("\n=== Duration (s) by strategy x model (multi-chunk docs only) ===")
        print(speed.round(1).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cron", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--project", default=PROJECT)
    args = p.parse_args()

    runner = Run2EvalRunner(
        run_matrix=RUN_MATRIX,
        dataset_loader=load_golden_dataset,
        judge_factory=make_gemini_judge,
        project=args.project,
        log_path=LOG_PATH,
        status_path=STATUS_PATH,
    )
    asyncio.run(runner.run(limit=args.limit, cron=args.cron, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test dry run**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python jobs/run2.py --dry-run
```

Expected: 14-entry matrix printed, exits cleanly with no import errors.

- [ ] **Step 3: Delete `evals/run2.py`**

```bash
git rm evals/run2.py
```

- [ ] **Step 4: Commit**

```bash
git add jobs/run2.py
git commit -m "refactor(evals): move run2 to jobs/ as thin EvalRunner caller, delete evals/run2.py"
```

---

### Task 4: Unit tests for ECW bin analysis

**Files:**
- Create: `evals/tests/test_effective_context_window.py`

The bin assignment and aggregation logic are pure functions. Test them before writing the script.

- [ ] **Step 1: Write failing tests**

```python
# evals/tests/test_effective_context_window.py
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "jobs"))

from effective_context_window import BINS, assign_bin, compute_degradation_curve


def test_assign_bin_first_bucket():
    assert assign_bin(1_000, BINS) == "<5K"


def test_assign_bin_boundary_at_5k_is_second_bucket():
    assert assign_bin(5_000, BINS) == "5K-12K"


def test_assign_bin_just_before_12k():
    assert assign_bin(11_999, BINS) == "5K-12K"


def test_assign_bin_at_12k_is_third_bucket():
    assert assign_bin(12_000, BINS) == "12K-30K"


def test_assign_bin_last_bucket():
    assert assign_bin(99_000, BINS) == "60K-100K"


def test_assign_bin_out_of_range_returns_none():
    assert assign_bin(200_000, BINS) is None


def test_assign_bin_zero():
    assert assign_bin(0, BINS) == "<5K"


def test_compute_curve_means():
    rows = [
        {"token_count": 1_000, "score": 0.9},
        {"token_count": 2_000, "score": 0.8},
        {"token_count": 8_000, "score": 0.7},
        {"token_count": 20_000, "score": 0.5},
    ]
    result = compute_degradation_curve(rows, BINS)
    assert result["<5K"]["mean"] == pytest.approx(0.85)
    assert result["5K-12K"]["mean"] == pytest.approx(0.7)
    assert result["12K-30K"]["mean"] == pytest.approx(0.5)
    assert result["30K-60K"]["n"] == 0
    assert result["30K-60K"]["mean"] is None


def test_compute_curve_counts():
    rows = [
        {"token_count": 1_000, "score": 0.9},
        {"token_count": 1_500, "score": 0.8},
        {"token_count": 6_000, "score": 0.7},
    ]
    result = compute_degradation_curve(rows, BINS)
    assert result["<5K"]["n"] == 2
    assert result["5K-12K"]["n"] == 1
    assert result["12K-30K"]["n"] == 0


def test_compute_curve_empty_rows():
    result = compute_degradation_curve([], BINS)
    for _, _, label in BINS:
        assert result[label]["n"] == 0
        assert result[label]["mean"] is None


def test_compute_curve_out_of_range_docs_ignored():
    rows = [{"token_count": 200_000, "score": 0.5}]
    result = compute_degradation_curve(rows, BINS)
    assert all(b["n"] == 0 for b in result.values())
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest evals/tests/test_effective_context_window.py -v
```

Expected: `ImportError` — `effective_context_window` not yet created.

---

### Task 5: Create `jobs/effective_context_window.py`

**Files:**
- Create: `jobs/effective_context_window.py`

- [ ] **Step 1: Write the script**

```python
# jobs/effective_context_window.py
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from dataset import ConduitDatasetAsync
from load_datasets import load_golden_dataset
from scorer import make_gemini_judge
from runner import EVAL_FUNCTION, EvalRunner
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer

BINS: list[tuple[int, int, str]] = [
    (0,       5_000,   "<5K"),
    (5_000,   12_000,  "5K-12K"),
    (12_000,  30_000,  "12K-30K"),
    (30_000,  60_000,  "30K-60K"),
    (60_000,  100_000, "60K-100K"),
]


def assign_bin(token_count: int, bins: list[tuple[int, int, str]]) -> str | None:
    for lo, hi, label in bins:
        if lo <= token_count < hi:
            return label
    return None


def compute_degradation_curve(
    rows: list[dict],
    bins: list[tuple[int, int, str]],
) -> dict[str, dict]:
    buckets: dict[str, list[float]] = {label: [] for _, _, label in bins}
    for row in rows:
        label = assign_bin(row["token_count"], bins)
        if label is not None:
            buckets[label].append(row["score"])
    out = {}
    for _, _, label in bins:
        scores = buckets[label]
        if scores:
            out[label] = {
                "n": len(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
            }
        else:
            out[label] = {"n": 0, "mean": None, "median": None}
    return out


class ECWEvalRunner(EvalRunner):
    def __init__(self, *args, results_path: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._results_path = results_path

    async def publish(self, ds: ConduitDatasetAsync, doc_meta: dict) -> None:
        eval_results = await ds.evals.list(eval_function=EVAL_FUNCTION)
        if not eval_results:
            print("No eval results in DB yet.")
            return

        rows = []
        for er in eval_results:
            r = er.run_result
            meta = doc_meta.get(r.source_id, {})
            rows.append({
                "source_id": r.source_id,
                "token_count": meta.get("token_count", 0),
                "score": er.score,
            })

        df = pd.DataFrame(rows)
        df.to_csv(self._results_path, index=False)
        print(f"\nResults saved to {self._results_path}")
        print(f"Total scored docs: {len(rows)}")

        curve = compute_degradation_curve(rows, BINS)
        print("\n=== Score degradation by token-length bin ===")
        print(f"{'Bin':<12} {'n':>5} {'mean':>7} {'median':>8}")
        print("-" * 36)
        for _, _, label in BINS:
            b = curve[label]
            mean_s  = f"{b['mean']:.3f}"   if b["mean"]   is not None else "  —  "
            med_s   = f"{b['median']:.3f}" if b["median"] is not None else "  —  "
            print(f"{label:<12} {b['n']:>5} {mean_s:>7} {med_s:>8}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure effective context window via summarization quality degradation"
    )
    p.add_argument("--model",      required=True, help="Model name (e.g. qwen3.6:latest)")
    p.add_argument("--server",     required=True, help="Server alias (e.g. deepwater, bywater)")
    p.add_argument("--token-cap",  type=int, default=100_000, help="Max token count per doc (default: 100000)")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--cron",       action="store_true")
    p.add_argument("--limit",      type=int)
    p.add_argument("--project",    help="DB project name (default: ecw_{model_slug})")
    args = p.parse_args()

    model_slug  = args.model.replace(":", "_").replace(".", "_")
    project     = args.project or f"ecw_{model_slug}"
    log_path    = Path(__file__).parent / f"{project}.log"
    status_path = Path(__file__).parent / f"{project}_status.json"
    results_path = Path(__file__).parent / f"{project}_results.csv"

    run_matrix = [{
        "strategy_cls": OneShotSummarizer,
        "config": {
            "model":      args.model,
            "use_remote": True,
            "host_alias": args.server,
            "use_cache":  True,
        },
        "server":          args.server,
        "timeout_s":       300,
        "concurrency":     3,
        "max_token_count": args.token_cap,
    }]

    runner = ECWEvalRunner(
        run_matrix=run_matrix,
        dataset_loader=load_golden_dataset,
        judge_factory=make_gemini_judge,
        project=project,
        log_path=log_path,
        status_path=status_path,
        results_path=results_path,
    )
    asyncio.run(runner.run(limit=args.limit, cron=args.cron, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run ECW unit tests**

```bash
python -m pytest evals/tests/test_effective_context_window.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 3: Smoke-test dry run**

```bash
python jobs/effective_context_window.py --model qwen3.6:latest --server deepwater --dry-run
```

Expected: single-entry matrix printed (`OneShotSummarizer / qwen3.6:latest / [deepwater]  cap=100K`), exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add jobs/effective_context_window.py evals/tests/test_effective_context_window.py
git commit -m "feat(jobs): add effective_context_window eval with score-by-bin degradation curve"
```

---

### Task 6: Create Cronicle event (HITL — manual step after deploy)

- [ ] **Step 1: Deploy to alphablue**

```bash
bash scripts/deploy.sh alphablue
```

- [ ] **Step 2: Create Cronicle event for the first model**

Replace `YOUR_API_KEY` with the key from Cronicle Administration → API Keys. Repeat this `curl` once per model you want to track, adjusting `--model`, `--server`, and the event `title`.

```bash
curl -s -X POST http://172.16.0.2:3012/api/app/create_event/v1 \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "ECW eval - qwen3.6:latest",
    "enabled": 1,
    "plugin": "shellplug",
    "target": "alphablue",
    "timing": {},
    "max_children": 1,
    "timeout": 0,
    "catch_up": 0,
    "params": {
      "script": "#!/bin/bash\nset -eo pipefail\nsource /home/fishhouses/.secrets\nsource /home/fishhouses/.exports\nexport XDG_DATA_HOME=/home/fishhouses/.local/share\nexport XDG_CONFIG_HOME=/home/fishhouses/.config\nexport XDG_STATE_HOME=/home/fishhouses/.local/state\ncd /home/fishhouses/Brian_Code/conduit-project\nexec /home/fishhouses/.local/bin/uv run python jobs/effective_context_window.py --model qwen3.6:latest --server deepwater --cron"
    }
  }'
```

- [ ] **Step 3: Update the existing run2 Cronicle event**

Find the run2 event in the Cronicle UI (Schedule tab). Edit the shell command: change `python evals/run2.py --cron` to `python jobs/run2.py --cron`. Save.

---

## Self-Review

**Spec coverage:**
- [x] Takes model + server as inputs → `--model`, `--server` in `main()` argparse
- [x] Runs OneShotSummarizer against 200-doc corpus, capped at 100K tokens → `max_token_count` in run matrix
- [x] Scores with Gemini judge → `_score_missing` calls `make_gemini_judge`
- [x] Outputs score by token-length bin → `compute_degradation_curve` + `publish()`
- [x] Publishable as on-demand Cronicle job, one event per model → Task 6
- [x] EvalRunner extracted first → Task 2
- [x] `run2.py` moved to `jobs/` as thin caller → Task 3
- [x] `evals/` holds abstractions, `jobs/` holds runnable entry points → file map

**Placeholder scan:** None found.

**Type consistency:** `assign_bin` returns `str | None` — used correctly in `compute_degradation_curve` (`if label is not None`). `ECWEvalRunner.__init__` takes `results_path: Path`, used in `publish()`. `EvalRunner.run()` signature `(*, limit, cron, dry_run)` matches all callers in `jobs/run2.py` and `jobs/effective_context_window.py`.
