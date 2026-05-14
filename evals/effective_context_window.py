from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from runner import EVAL_FUNCTION, EvalRunner

if TYPE_CHECKING:
    from dataset import ConduitDatasetAsync

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
