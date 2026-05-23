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
from conduit.strategies.summarize.summarizers.map_dedupe_reduce import (
    MapDedupeReduceHybridModelSummarizer,
    MapDedupeReduceSummarizer,
)
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

# Hybrid: gpt-oss for cheap parallel chunk extraction, gemma4 for dedupe and final reduce.
# use_cache=False to guarantee fresh results for the new hybrid config.
_HYBRID_GPT_CHUNKS_GEMMA_REDUCE = {
    "model":             "gemma4:latest",
    "use_remote":        True,
    "host_alias":        "deepwater",
    "use_cache":         False,
    "chunk_model":       "gpt-oss:latest",
    "chunk_host_alias":  "bywater",
    "dedupe_model":      "gemma4:latest",
    "dedupe_host_alias": "deepwater",
    "reduce_model":      "gemma4:latest",
    "reduce_host_alias": "deepwater",
}

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
    {
        "strategy_cls": MapDedupeReduceHybridModelSummarizer,
        "config":       _HYBRID_GPT_CHUNKS_GEMMA_REDUCE,
        "server":       "deepwater",
        "timeout_s":    1800,
        "concurrency":  1,
    },
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
