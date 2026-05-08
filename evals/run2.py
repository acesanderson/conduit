"""
Run 2: Strategy comparison runner.

Tests RollingRefineSummarizer, MapDedupeReduceSummarizer, HierarchicalTreeSummarizer
against the RecursiveSummarizer baseline × qwen3.6 (deepwater), gpt-oss (bywater),
and gemma4 (deepwater).

Resumable: skips (strategy, config) pairs that are already fully complete in DB.
Smoke gate: runs 2 docs before each new (strategy, server) combination.

Usage:
    python evals/run2.py               # full run
    python evals/run2.py --dry-run     # print matrix and exit
    python evals/run2.py --cron        # ping servers first; exit 0 if unreachable
    python evals/run2.py --limit 10    # first N docs (quick test)
    python evals/run2.py --project X   # override project name

Cron schedule:
    0 2 * * * cd /Users/bianders/Brian_Code/conduit-project && python evals/run2.py --cron >> evals/run2_cron.log 2>&1

Diagnostics (all in this file):
    - run_failures table: persists every failure with error_type, token_count, traceback
    - ServerCircuitBreaker: per-server open/reset logic to stop hammering crashed Ollama
    - Enriched logging: source_id, token_count, inflight count, full traceback on error

Query failures after a run:
    SELECT error_type, COUNT(*), AVG(token_count)
    FROM run_failures WHERE project = 'run2_strategy_comparison'
    GROUP BY error_type ORDER BY count DESC;
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from dataset import ConduitDatasetAsync
from evals import (
    CONCURRENCY_LIMIT,
    EvalResult,
    RunInput,
    RunResult,
    generate_runs,
    run_eval,
)
from load_datasets import load_golden_dataset
from scorer import make_gemini_judge
from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer
from conduit.strategies.summarize.summarizers.rolling_refine import RollingRefineSummarizer
from conduit.strategies.summarize.summarizers.map_dedupe_reduce import MapDedupeReduceSummarizer
from conduit.strategies.summarize.summarizers.hierarchical_tree import HierarchicalTreeSummarizer

PROJECT = "run2_strategy_comparison"
EVAL_FUNCTION = "gemini_judge"
RESULTS_PATH = Path(__file__).parent / "results_run2.csv"
LOG_PATH = Path(__file__).parent / "run2.log"

_QWEN = {"model": "qwen3.6:latest", "use_remote": True, "host_alias": "deepwater", "use_cache": True}
_QWEN_RECURSIVE = {**_QWEN, "map_model": "gpt-oss:latest", "map_host_alias": "bywater"}
_GPT = {"model": "gpt-oss:latest", "use_remote": True, "host_alias": "bywater", "use_cache": True}
_GEMMA = {"model": "gemma4:latest", "use_remote": True, "host_alias": "deepwater", "use_cache": True}
_GEMMA_RECURSIVE = {**_GEMMA, "map_model": "gpt-oss:latest", "map_host_alias": "bywater"}

RUN_MATRIX = [
    # Deepwater (qwen3.6) — larger model, slower per call; run 1 doc at a time for
    # multi-stage strategies so Ollama never sees more than ~5 concurrent requests.
    {"strategy_cls": RecursiveSummarizer,        "config": _QWEN_RECURSIVE, "server": "deepwater", "timeout_s": 600,  "concurrency": 5},
    {"strategy_cls": RollingRefineSummarizer,    "config": _QWEN,           "server": "deepwater", "timeout_s": 3600, "concurrency": 1},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _QWEN,           "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _QWEN,           "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    # Bywater (gpt-oss) — faster per call; allow 2 concurrent docs for parallel strategies.
    {"strategy_cls": RecursiveSummarizer,        "config": _GPT,            "server": "bywater",   "timeout_s": 600,  "concurrency": 5},
    {"strategy_cls": RollingRefineSummarizer,    "config": _GPT,            "server": "bywater",   "timeout_s": 2400, "concurrency": 1},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _GPT,            "server": "bywater",   "timeout_s": 1200, "concurrency": 2},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _GPT,            "server": "bywater",   "timeout_s": 1200, "concurrency": 2},
    # Deepwater (gemma4) — same concurrency constraints as qwen3.6.
    {"strategy_cls": RecursiveSummarizer,        "config": _GEMMA_RECURSIVE, "server": "deepwater", "timeout_s": 600,  "concurrency": 5},
    {"strategy_cls": RollingRefineSummarizer,    "config": _GEMMA,           "server": "deepwater", "timeout_s": 3600, "concurrency": 1},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _GEMMA,           "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _GEMMA,           "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure persistence
# ---------------------------------------------------------------------------

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
CREATE INDEX IF NOT EXISTS idx_run_failures_project  ON run_failures (project);
CREATE INDEX IF NOT EXISTS idx_run_failures_error    ON run_failures (error_type);
CREATE INDEX IF NOT EXISTS idx_run_failures_source   ON run_failures (source_id);
"""


async def _ensure_failures_table() -> None:
    from persist import _get_pool
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_FAILURES_DDL)


async def _save_failure(
    project: str,
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
                project, strategy, config_id, source_id,
                error_type, error_message[:500], token_count, tb_str[:4000],
            )
    except Exception as db_exc:
        logger.warning("failed to persist failure record: %s", db_exc)


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


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class ServerCircuitBreaker:
    """
    Opens after `threshold` consecutive failures; blocks new dispatches until
    `cooldown_s` has elapsed. All tasks call wait_if_open() before each request.

    Distinguishes sustained server-down (OOM/crash cascade) from transient blips:
    a blip recovers on its own; a crash produces threshold+ consecutive failures.
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


def _config_id(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def setup_logging() -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Print matrix and exit")
    p.add_argument("--cron", action="store_true", help="Ping servers first; exit 0 if unreachable")
    p.add_argument("--limit", type=int, help="Cap number of docs (quick test)")
    p.add_argument("--project", default=PROJECT, help="DB project name")
    return p.parse_args()


async def warmup_server(alias: str) -> bool:
    from headwater_api.classes import BatchRequest
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.utils.progress.verbosity import Verbosity

    model = "qwen3.6:latest" if alias == "deepwater" else "gpt-oss:latest"
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


async def health_check() -> bool:
    for alias in ("deepwater", "bywater"):
        try:
            async with HeadwaterAsyncClient(host_alias=alias) as client:
                ok = await client.ping()
            if not ok:
                print(f"[cron] {alias} ping returned False — aborting")
                return False
        except Exception as exc:
            print(f"[cron] {alias} unreachable: {exc} — aborting")
            return False

        if not await warmup_server(alias):
            print(f"[cron] {alias} warmup failed — aborting")
            return False

    return True


async def get_done_ids(ds: ConduitDatasetAsync, strategy_name: str, cid: str) -> set[str]:
    results = await ds.runs.list(strategy=strategy_name, config_id=cid)
    return {r.source_id for r in results}


async def _run_inference_incremental(
    docs: list[RunInput],
    config: dict,
    strategy,
    ds: ConduitDatasetAsync,
    timeout_s: int,
    circuit_breaker: ServerCircuitBreaker,
    project: str,
    concurrency: int = CONCURRENCY_LIMIT,
) -> list[RunResult]:
    sem = asyncio.Semaphore(concurrency)
    strategy_name = strategy.__class__.__name__
    cid = _config_id(config)
    inflight: list[int] = [0]  # mutable cell; asyncio is single-threaded so no lock needed

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
                await _save_failure(project, strategy_name, cid, doc.source_id,
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
                await _save_failure(project, strategy_name, cid, doc.source_id,
                                    error_type, str(exc), token_count, tb_str)
                return None
            finally:
                inflight[0] -= 1

    tasks = [asyncio.create_task(run_and_save(doc)) for doc in docs]
    raw = await asyncio.gather(*tasks)
    return [r for r in raw if r is not None]


async def score_missing(
    ds: ConduitDatasetAsync,
    strategy_name: str,
    cid: str,
    judge,
) -> list[EvalResult]:
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
                    run_result.source_id,
                    type(exc).__name__,
                    exc,
                )
                return None

    tasks = [asyncio.create_task(score_and_save(r)) for r in unscored]
    raw = await asyncio.gather(*tasks)
    return [r for r in raw if r is not None]


async def run_entry(
    ds: ConduitDatasetAsync,
    entry: dict,
    docs: list[RunInput],
    judge,
    smoke_tested: set[tuple[str, str]],
    circuit_breaker: ServerCircuitBreaker,
    project: str,
) -> list[RunResult]:
    strategy = entry["strategy_cls"]()
    config = entry["config"]
    strategy_name = strategy.__class__.__name__
    server = entry["server"]
    cid = _config_id(config)
    n_total = len(docs)

    done_ids = await get_done_ids(ds, strategy_name, cid)
    if len(done_ids) >= n_total:
        await score_missing(ds, strategy_name, cid, judge)
        return []

    remaining = [d for d in docs if d.source_id not in done_ids]
    if done_ids:
        print(f"  RESUME {strategy_name}/{cid}: {len(done_ids)}/{n_total} done, {len(remaining)} remaining")
    else:
        print(f"  START  {strategy_name}/{cid} × {n_total} docs  [{server}]")

    # Smoke test: run 2 docs before committing to a full batch for any new (strategy, server)
    smoke_key = (strategy_name, server)
    if not done_ids and smoke_key not in smoke_tested:
        print(f"  Smoke: {strategy_name} on {server} × 2 docs ...")
        try:
            smoke_results = await generate_runs(inputs=remaining[:2], configs=[config], strategy=strategy)
            if len(smoke_results) < 2:
                print(f"  ABORT  {strategy_name}/{server} — smoke test failed ({len(smoke_results)}/2 succeeded)")
                return []
            print(f"  Smoke passed.")
            smoke_tested.add(smoke_key)
        except Exception as exc:
            print(f"  ABORT  {strategy_name}/{server} — smoke test error: {exc}")
            logger.exception("smoke test error strategy=%s server=%s", strategy_name, server)
            return []

    run_results = await _run_inference_incremental(
        docs=remaining,
        config=config,
        strategy=strategy,
        ds=ds,
        timeout_s=entry["timeout_s"],
        circuit_breaker=circuit_breaker,
        project=project,
        concurrency=entry.get("concurrency", CONCURRENCY_LIMIT),
    )
    print(f"  Done.  {strategy_name}/{cid}: {len(run_results)}/{len(remaining)} succeeded.")

    await score_missing(ds, strategy_name, cid, judge)

    return run_results


async def print_results(ds: ConduitDatasetAsync, doc_meta: dict) -> None:
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

    # Analysis excludes single-chunk docs — all strategies produce identical output
    # for docs that fit in one chunk, so they add noise without signal.
    CHUNK_SIZE = 12000
    df_multi = df[df["token_count"] > CHUNK_SIZE].copy()
    n_one_shot = len(df) - len(df_multi)
    n_strategies = df["strategy"].nunique()
    print(f"\nAnalysis: excluding {n_one_shot // n_strategies} one-shot docs (token_count ≤ {CHUNK_SIZE}), "
          f"using {len(df_multi) // n_strategies} multi-chunk docs")

    summary = (
        df_multi.groupby(["strategy", "model"])["score"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"count": "n"})
        .sort_values("mean", ascending=False)
    )
    print("\n=== Scores by strategy × model (multi-chunk docs only) ===")
    print(summary.round(3).to_string())

    by_category = df_multi.groupby(["strategy", "category"])["score"].mean().unstack("category")
    print("\n=== Scores by strategy × category (multi-chunk docs only) ===")
    print(by_category.round(3).to_string())

    speed = (
        df_multi.dropna(subset=["duration_s"])
        .groupby(["strategy", "model"])["duration_s"]
        .agg(["mean", "median", "max"])
        .sort_values("mean")
    )
    print("\n=== Duration (s) by strategy × model (multi-chunk docs only) ===")
    print(speed.round(1).to_string())


STATUS_PATH = Path(__file__).parent / "run2_status.json"


def _write_status(status: dict) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))


def _notify(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


async def main() -> None:
    args = parse_args()
    setup_logging()

    if args.cron:
        if not await health_check():
            sys.exit(0)

    docs = load_golden_dataset()
    if args.limit:
        docs = docs[: args.limit]

    deepwater_entries = [e for e in RUN_MATRIX if e["server"] == "deepwater"]
    bywater_entries = [e for e in RUN_MATRIX if e["server"] == "bywater"]

    print(f"Project:   {args.project}")
    print(f"Docs:      {len(docs)}")
    print(f"Matrix ({len(RUN_MATRIX)} entries):")
    for e in RUN_MATRIX:
        print(f"  {e['strategy_cls'].__name__:<30} {e['config'].get('model'):<20} [{e['server']}]")

    if args.dry_run:
        return

    ds = ConduitDatasetAsync(args.project)

    print("\nSeeding documents to DB...")
    await ds.documents.save(docs)
    await _ensure_failures_table()

    references = {doc.source_id: doc.reference for doc in docs}
    judge = make_gemini_judge(references)
    doc_meta = {doc.source_id: doc.metadata for doc in docs}

    smoke_tested: set[tuple[str, str]] = set()
    all_results: list[RunResult] = []
    started_at = datetime.now()

    circuit_breakers = {
        "deepwater": ServerCircuitBreaker("deepwater", threshold=5, cooldown_s=60.0),
        "bywater":   ServerCircuitBreaker("bywater",   threshold=5, cooldown_s=60.0),
    }

    try:
        async def run_bywater() -> list[RunResult]:
            results = []
            for entry in bywater_entries:
                batch = await run_entry(
                    ds, entry, docs, judge, smoke_tested,
                    circuit_breaker=circuit_breakers["bywater"],
                    project=args.project,
                )
                results.extend(batch)
            return results

        print("\nRunning summarizations:")
        bywater_task = asyncio.create_task(run_bywater())

        for entry in deepwater_entries:
            batch = await run_entry(
                ds, entry, docs, judge, smoke_tested,
                circuit_breaker=circuit_breakers["deepwater"],
                project=args.project,
            )
            all_results.extend(batch)

        bywater_results = await bywater_task
        all_results.extend(bywater_results)

        print(f"\nTotal new results this run: {len(all_results)}")

        await print_results(ds, doc_meta)

        _write_status({
            "result": "ok",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "new_results": len(all_results),
        })
        _notify("run2 complete", f"{len(all_results)} new results")

    except Exception as exc:
        _write_status({
            "result": "failed",
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        _notify("run2 FAILED", str(exc)[:100])
        raise


if __name__ == "__main__":
    asyncio.run(main())
