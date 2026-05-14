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

    model_slug   = args.model.replace(":", "_").replace(".", "_")
    project      = args.project or f"ecw_{model_slug}"
    log_path     = Path(__file__).parent / f"{project}.log"
    status_path  = Path(__file__).parent / f"{project}_status.json"
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
