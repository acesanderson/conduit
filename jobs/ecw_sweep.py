"""
ecw_sweep - Effective Context Window eval across multiple models.

Runs OneShotSummarizer over the 200-doc gold dataset to measure quality
degradation by token count for each model. Models that exist on both
bywater and deepwater have their docs split deterministically across both
hosts (even/odd by source_id hex hash) for cross-host parallelism.

Resumable across nights: each run skips (strategy, config_id, source_id)
triples already in the DB.

Usage:
    uv run python jobs/ecw_sweep.py                # full sweep
    uv run python jobs/ecw_sweep.py --cron         # health-gated (for Cronicle)
    uv run python jobs/ecw_sweep.py --dry-run      # print matrix, exit
    uv run python jobs/ecw_sweep.py --limit 4      # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

import pandas as pd

from effective_context_window import BINS, ECWEvalRunner, compute_degradation_curve
from load_datasets import load_golden_dataset
from runner import EVAL_FUNCTION
from scorer import make_gemini_judge
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer

if TYPE_CHECKING:
    from dataset import ConduitDatasetAsync
    from evals import RunInput

PROJECT = "ecw_sweep"
TOKEN_CAP = 150_000
LOG_PATH = Path(__file__).parent / f"{PROJECT}.log"
STATUS_PATH = Path(__file__).parent / f"{PROJECT}_status.json"
RESULTS_PATH = Path(__file__).parent / f"{PROJECT}_results.csv"


def even_partition(doc: "RunInput") -> bool:
    return int(doc.source_id, 16) % 2 == 0


def odd_partition(doc: "RunInput") -> bool:
    return int(doc.source_id, 16) % 2 == 1


def _entry(model: str, host: str, predicate=None, concurrency: int = 3) -> dict:
    return {
        "strategy_cls": OneShotSummarizer,
        "config": {
            "model":      model,
            "use_remote": True,
            "host_alias": host,
            "use_cache":  True,
        },
        "server":          host,
        "timeout_s":       300,
        "concurrency":     concurrency,
        "max_token_count": TOKEN_CAP,
        "doc_predicate":   predicate,
    }


# Caruana-friendly models (present in both ollama configs): split across hosts.
SHARED_MODELS = [
    "gpt-oss:latest",
    "cogito:32b",
    "qwen3:30b",
]

# Alphablue-only models: deepwater only.
DEEPWATER_ONLY_MODELS = [
    "qwen3.6:latest",
    "gemma4:latest",
]

RUN_MATRIX: list[dict] = []
for m in SHARED_MODELS:
    RUN_MATRIX.append(_entry(m, "bywater",   predicate=even_partition))
    RUN_MATRIX.append(_entry(m, "deepwater", predicate=odd_partition))
for m in DEEPWATER_ONLY_MODELS:
    RUN_MATRIX.append(_entry(m, "deepwater"))


class SweepECWEvalRunner(ECWEvalRunner):
    """ECW runner that publishes per-model degradation curves."""

    async def publish(self, ds: "ConduitDatasetAsync", doc_meta: dict) -> None:
        eval_results = await ds.evals.list(eval_function=EVAL_FUNCTION)
        if not eval_results:
            print("No eval results in DB yet.")
            return

        rows = []
        for er in eval_results:
            r = er.run_result
            meta = doc_meta.get(r.source_id, {})
            config = r.config if isinstance(r.config, dict) else r.config.model_dump()
            rows.append({
                "model":       config.get("model", ""),
                "host_alias":  config.get("host_alias", ""),
                "source_id":   r.source_id,
                "token_count": meta.get("token_count", 0),
                "score":       er.score,
            })

        df = pd.DataFrame(rows)
        df.to_csv(self._results_path, index=False)
        print(f"\nResults saved to {self._results_path}")
        print(f"Total scored runs: {len(rows)}  ({df['model'].nunique()} models)")

        for model in sorted(df["model"].unique()):
            model_rows = df[df["model"] == model].to_dict("records")
            curve = compute_degradation_curve(model_rows, BINS)
            print(f"\n=== {model} ({len(model_rows)} docs) ===")
            print(f"{'Bin':<12} {'n':>5} {'mean':>7} {'median':>8}")
            print("-" * 36)
            for _, _, label in BINS:
                b = curve[label]
                mean_s = f"{b['mean']:.3f}"   if b["mean"]   is not None else "  —  "
                med_s  = f"{b['median']:.3f}" if b["median"] is not None else "  —  "
                print(f"{label:<12} {b['n']:>5} {mean_s:>7} {med_s:>8}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cron",    action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit",   type=int)
    p.add_argument("--project", default=PROJECT)
    args = p.parse_args()

    runner = SweepECWEvalRunner(
        run_matrix=RUN_MATRIX,
        dataset_loader=load_golden_dataset,
        judge_factory=make_gemini_judge,
        project=args.project,
        log_path=LOG_PATH,
        status_path=STATUS_PATH,
        results_path=RESULTS_PATH,
    )
    asyncio.run(runner.run(limit=args.limit, cron=args.cron, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
