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
from conduit.strategies.summarize.summarizers.map_reduce import MapReduceSummarizer
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer
from conduit.strategies.summarize.summarizers.rolling_refine import RollingRefineSummarizer

PROJECT = "run2_strategy_comparison"
RESULTS_PATH = Path(__file__).parent / "run2_results.csv"
LOG_PATH = Path(__file__).parent / "run2.log"
STATUS_PATH = Path(__file__).parent / "run2_status.json"

# Rerun matrix — qwen3.6 dropped (out of latency budget for production routing).
# All entries use_cache=False to force fresh runs with clean duration capture.
# Goal: validate the three-tier routing (gpt-oss/OneShot, gemma4/OneShot, gemma4/RollingRefine)
# and measure the unmeasured MapReduce candidates for the chunked tier.

_GPT   = {"model": "gpt-oss:latest", "use_remote": True, "host_alias": "bywater",   "use_cache": False}
_GEMMA = {"model": "gemma4:latest",  "use_remote": True, "host_alias": "deepwater", "use_cache": False}

# Hybrid: gpt-oss for cheap parallel chunk extraction (bywater) + gemma4 for dedupe and final reduce (deepwater).
_HYBRID = {
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
    # Tier 1 — OneShot + gpt-oss on small docs. Cap at 12K (one bin past gpt-oss's ECW cliff).
    {"strategy_cls": OneShotSummarizer, "config": _GPT,   "server": "bywater",   "timeout_s": 300, "concurrency": 5, "max_token_count": 12_000},

    # Tier 2 — OneShot + gemma4 across the gemma4-viable range. Cap at 60K (one bin past gemma4's ECW cliff).
    {"strategy_cls": OneShotSummarizer, "config": _GEMMA, "server": "deepwater", "timeout_s": 300, "concurrency": 3, "max_token_count": 60_000},

    # Tier 3 baseline — RollingRefine + gemma4. Current winner; rerun captures real wall-clock.
    {"strategy_cls": RollingRefineSummarizer, "config": _GEMMA, "server": "deepwater", "timeout_s": 1200, "concurrency": 1},

    # Tier 3 candidate A — MapDedupeReduceHybrid. Cross-host: gpt-oss chunks on bywater + gemma4 reduce on deepwater.
    {"strategy_cls": MapDedupeReduceHybridModelSummarizer, "config": _HYBRID, "server": "deepwater", "timeout_s": 1800, "concurrency": 1},

    # Tier 3 candidate B — plain MapReduce + gemma4. Isolates whether the dedupe pass is the cost driver vs MDR.
    {"strategy_cls": MapReduceSummarizer, "config": _GEMMA, "server": "deepwater", "timeout_s": 1800, "concurrency": 1},
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
            # Sum across all trace entries — trace[-1] only captures the final call,
            # which understates wall-clock for chunked strategies (Rolling, MapReduce, etc.).
            duration = sum(t.get("duration", 0) for t in trace) if trace else None
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
