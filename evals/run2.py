"""
Run 2: Strategy comparison runner.

Tests RollingRefineSummarizer, MapDedupeReduceSummarizer, HierarchicalTreeSummarizer
against the RecursiveSummarizer baseline × qwen3.6 (deepwater) and gpt-oss (bywater).

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
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from dataset import BatchSaveError, ConduitDatasetAsync
from evals import (
    CONCURRENCY_LIMIT,
    EvalResult,
    RunInput,
    RunResult,
    evaluate,
    generate_runs,
    run_eval,
)
from load_datasets import load_golden_dataset
from scorer import make_gemini_judge
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

RUN_MATRIX = [
    # Deepwater (qwen3.6)
    {"strategy_cls": RecursiveSummarizer,        "config": _QWEN_RECURSIVE, "server": "deepwater", "timeout_s": 600},
    {"strategy_cls": RollingRefineSummarizer,    "config": _QWEN,           "server": "deepwater", "timeout_s": 1800},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _QWEN,           "server": "deepwater", "timeout_s": 900},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _QWEN,           "server": "deepwater", "timeout_s": 900},
    # Bywater (gpt-oss)
    {"strategy_cls": RecursiveSummarizer,        "config": _GPT,            "server": "bywater",   "timeout_s": 600},
    {"strategy_cls": RollingRefineSummarizer,    "config": _GPT,            "server": "bywater",   "timeout_s": 1800},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _GPT,            "server": "bywater",   "timeout_s": 900},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _GPT,            "server": "bywater",   "timeout_s": 900},
]

logger = logging.getLogger(__name__)


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


async def ping_servers() -> bool:
    from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
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
) -> list[RunResult]:
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_and_save(doc: RunInput) -> RunResult | None:
        async with sem:
            try:
                result = await asyncio.wait_for(
                    run_eval(doc, config, strategy), timeout=timeout_s
                )
                await ds.runs.save([result])
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "timeout source_id=%s after %ds", doc.source_id, timeout_s
                )
                return None
            except Exception as exc:
                logger.error(
                    "run_eval failed source_id=%s: %s: %s",
                    doc.source_id,
                    type(exc).__name__,
                    exc,
                )
                return None

    tasks = [asyncio.create_task(run_and_save(doc)) for doc in docs]
    raw = await asyncio.gather(*tasks)
    return [r for r in raw if r is not None]


async def run_entry(
    ds: ConduitDatasetAsync,
    entry: dict,
    docs: list[RunInput],
    judge,
    smoke_tested: set[tuple[str, str]],
) -> list[RunResult]:
    strategy = entry["strategy_cls"]()
    config = entry["config"]
    strategy_name = strategy.__class__.__name__
    server = entry["server"]
    cid = _config_id(config)
    n_total = len(docs)

    done_ids = await get_done_ids(ds, strategy_name, cid)
    if len(done_ids) >= n_total:
        # Runs complete — check whether evals were also saved
        existing_evals = await ds.evals.list(strategy=strategy_name, config_id=cid)
        scored_ids = {er.run_result.source_id for er in existing_evals}
        unscored = [r for r in await ds.runs.list(strategy=strategy_name, config_id=cid)
                    if r.source_id not in scored_ids]
        if not unscored:
            print(f"  SKIP  {strategy_name}/{cid} — complete ({len(done_ids)} runs, {len(existing_evals)} evals)")
            return []
        print(f"  RESCORE {strategy_name}/{cid} — {len(unscored)} evals missing")
        eval_results = await evaluate(unscored, eval_function=judge)
        try:
            await ds.evals.save(eval_results, eval_function=EVAL_FUNCTION)
        except BatchSaveError as exc:
            print(f"  Warning: partial eval save — {exc}")
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

    run_results = await generate_runs(inputs=remaining, configs=[config], strategy=strategy)
    print(f"  Done.  {strategy_name}/{cid}: {len(run_results)}/{len(remaining)} succeeded.")

    if run_results:
        await ds.runs.save(run_results)
        eval_results = await evaluate(run_results, eval_function=judge)
        try:
            await ds.evals.save(eval_results, eval_function=EVAL_FUNCTION)
        except BatchSaveError as exc:
            print(f"  Warning: partial eval save failure — {exc}")

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
        duration = trace[0]["duration"] if trace else None
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


async def main() -> None:
    args = parse_args()
    setup_logging()

    if args.cron:
        if not await ping_servers():
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

    references = {doc.source_id: doc.reference for doc in docs}
    judge = make_gemini_judge(references)
    doc_meta = {doc.source_id: doc.metadata for doc in docs}

    smoke_tested: set[tuple[str, str]] = set()
    all_results: list[RunResult] = []

    async def run_bywater() -> list[RunResult]:
        results = []
        for entry in bywater_entries:
            batch = await run_entry(ds, entry, docs, judge, smoke_tested)
            results.extend(batch)
        return results

    print("\nRunning summarizations:")
    bywater_task = asyncio.create_task(run_bywater())

    for entry in deepwater_entries:
        batch = await run_entry(ds, entry, docs, judge, smoke_tested)
        all_results.extend(batch)

    bywater_results = await bywater_task
    all_results.extend(bywater_results)

    print(f"\nTotal new results this run: {len(all_results)}")

    await print_results(ds, doc_meta)


if __name__ == "__main__":
    asyncio.run(main())
