"""
Hybrid eval run: RecursiveSummarizer with split model routing.
  - One-shot (doc fits in context): gemma4:latest on deepwater
  - Chunk summaries (map phase):    gpt-oss:latest on bywater

Compares against the gemma4-only baseline in 'recursive_summarization'.

Usage:
    python run_hybrid.py
    python run_hybrid.py --dry-run
    python run_hybrid.py --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from dataset import BatchSaveError, ConduitDatasetAsync
from evals import generate_runs, evaluate
from load_datasets import load_golden_dataset
from scorer import make_gemini_judge
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer

PROJECT = "hybrid_summarization"
EVAL_FUNCTION = "gemini_judge"
RESULTS_PATH = Path(__file__).parent / "results_hybrid.csv"

CONFIG = {
    "model": "gemma4:latest",
    "host_alias": "deepwater",
    "map_model": "gpt-oss:latest",
    "map_host_alias": "bywater",
    "use_remote": True,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int)
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    docs = load_golden_dataset()
    if args.limit:
        docs = docs[: args.limit]

    print(f"Project:     {PROJECT}")
    print(f"One-shot:    {CONFIG['model']} on {CONFIG['host_alias']}")
    print(f"Chunk/map:   {CONFIG['map_model']} on {CONFIG['map_host_alias']}")
    print(f"Docs:        {len(docs)}")

    if args.dry_run:
        return

    ds = ConduitDatasetAsync(PROJECT)

    print("\nSeeding documents to DB...")
    await ds.documents.save(docs)

    strategy = RecursiveSummarizer()
    references = {doc.source_id: doc.reference for doc in docs}
    judge = make_gemini_judge(references)
    doc_meta = {doc.source_id: doc.metadata for doc in docs}

    print(f"\nRunning {len(docs)} docs ...")
    run_results = await generate_runs(inputs=docs, configs=[CONFIG], strategy=strategy)
    print(f"Done. {len(run_results)} results.")

    print("Persisting run results to DB...")
    await ds.runs.save(run_results)

    print("Scoring with Gemini3 judge...")
    eval_results = await evaluate(run_results, eval_function=judge)
    print(f"Done. {len(eval_results)} scores.")

    print("Persisting eval results to DB...")
    try:
        await ds.evals.save(eval_results, eval_function=EVAL_FUNCTION)
    except BatchSaveError as exc:
        print(f"Warning: {exc} — continuing to CSV.")

    rows = []
    for er in eval_results:
        r = er.run_result
        meta = doc_meta.get(r.source_id, {})
        config = r.config if isinstance(r.config, dict) else r.config.model_dump()
        trace = r.output.metadata.get("trace", [])
        duration = trace[0]["duration"] if trace else None
        rows.append({
            "source_id": r.source_id,
            "oneshot_model": config.get("model", ""),
            "map_model": config.get("map_model", config.get("model", "")),
            "category": meta.get("category", ""),
            "token_count": meta.get("token_count", 0),
            "score": er.score,
            "output_chars": len(r.output.output),
            "duration_s": duration,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    summary = df["score"].agg(["mean", "median", "std", "count"]).rename({"count": "n"})
    print("\n=== Hybrid scores ===")
    print(summary.round(3).to_string())

    by_category = df.groupby("category")["score"].agg(["mean", "count"]).rename(columns={"count": "n"})
    print("\n=== Scores by category ===")
    print(by_category.round(3).sort_values("mean", ascending=False).to_string())

    speed = df["duration_s"].agg(["mean", "median", "max"])
    print("\n=== Duration (s) ===")
    print(speed.round(1).to_string())


if __name__ == "__main__":
    asyncio.run(main())
