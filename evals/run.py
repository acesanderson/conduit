"""
Eval runner: RecursiveSummarizer × 4 local models × 200 gold-standard docs.
Scores each output against Gemini3 reference summaries using LLM-as-judge.
Results are persisted to the `evals` Postgres database and saved as CSV.

Usage:
    cd evals/
    python run.py                          # full run, all 4 models
    python run.py --models qwen3.6         # fuzzy-match subset
    python run.py --dry-run                # print config and exit
    python run.py --limit 10               # first N docs only (smoke test)
    python run.py --project my-experiment  # custom project name
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

MODELS = [
    "gpt-oss:latest",
    "qwen3.6:latest",
    "gemma4:latest",
]

EVAL_FUNCTION = "gemini_judge"
RESULTS_PATH = Path(__file__).parent / "results.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", help="Model name substrings to include")
    p.add_argument("--limit", type=int, help="Cap number of docs (for smoke tests)")
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    p.add_argument("--project", default="recursive_summarization", help="Project name for DB namespacing")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    models = MODELS
    if args.models:
        models = [m for m in MODELS if any(s in m for s in args.models)]
        if not models:
            print(f"No models matched {args.models}. Available: {MODELS}")
            sys.exit(1)

    docs = load_golden_dataset()
    if args.limit:
        docs = docs[: args.limit]

    configs = [{"model": m, "use_remote": True} for m in models]
    n_runs = len(docs) * len(configs)

    print(f"Project: {args.project}")
    print(f"Docs: {len(docs)}  |  Models: {len(configs)}  |  Total runs: {n_runs}")
    for m in models:
        print(f"  {m}")

    if args.dry_run:
        return

    ds = ConduitDatasetAsync(args.project)

    # Seed documents (idempotent — ON CONFLICT skips existing rows)
    print("\nSeeding documents to DB...")
    await ds.documents.save(docs)
    print(f"  {len(docs)} documents ready.")

    strategy = RecursiveSummarizer()
    references = {doc.source_id: doc.reference for doc in docs}
    judge = make_gemini_judge(references)
    doc_meta = {doc.source_id: doc.metadata for doc in docs}

    # qwen3.6 and gemma4 only exist on deepwater; command-r and gpt-oss are on both.
    # Each server owns its models exclusively — no splitting, no switching.
    DEEPWATER_MODELS = {"qwen3.6:latest", "gemma4:latest"}
    BYWATER_MODELS = {"gpt-oss:latest"}
    deep_models = [m for m in models if m in DEEPWATER_MODELS]
    by_models = [m for m in models if m in BYWATER_MODELS]

    print(f"\nRunning summarizations:")
    run_results = []

    # Bywater: gpt-oss × all 200 docs, starts immediately in background.
    async def run_bywater() -> list:
        results = []
        for m in by_models:
            cfg = [{"model": m, "use_remote": True, "host_alias": "bywater"}]
            print(f"  bywater: {m} × {len(docs)} docs  [starting now]")
            batch = await generate_runs(inputs=docs, configs=cfg, strategy=strategy)
            results.extend(batch)
        return results

    bywater_task = asyncio.create_task(run_bywater()) if by_models else None

    # Deepwater: qwen3.6 then gemma4, sequentially — one model in VRAM at a time.
    # Sleep between models so Ollama has time to evict the previous model before
    # the next model's health check fires.
    for i, m in enumerate(deep_models):
        if i > 0:
            print(f"  Waiting 30s for Ollama to swap models...")
            await asyncio.sleep(30)
        cfg = [{"model": m, "use_remote": True, "host_alias": "deepwater"}]
        print(f"  deepwater: {m} × {len(docs)} docs")
        batch = await generate_runs(inputs=docs, configs=cfg, strategy=strategy)
        run_results.extend(batch)

    if bywater_task is not None:
        bywater_results = await bywater_task
        run_results.extend(bywater_results)
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

    # CSV for quick inspection
    rows = []
    for er in eval_results:
        r = er.run_result
        meta = doc_meta.get(r.source_id, {})
        config = r.config if isinstance(r.config, dict) else r.config.model_dump()
        trace = r.output.metadata.get("trace", [])
        duration = trace[0]["duration"] if trace else None
        rows.append({
            "source_id": r.source_id,
            "model": config.get("model", ""),
            "category": meta.get("category", ""),
            "token_count": meta.get("token_count", 0),
            "score": er.score,
            "output_chars": len(r.output.output),
            "duration_s": duration,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults also saved to {RESULTS_PATH}")

    summary = (
        df.groupby("model")["score"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"count": "n"})
        .sort_values("mean", ascending=False)
    )
    print("\n=== Scores by model ===")
    print(summary.round(3).to_string())

    by_category = df.groupby(["model", "category"])["score"].mean().unstack("category")
    print("\n=== Scores by model × category ===")
    print(by_category.round(3).to_string())

    speed = (
        df.groupby("model")["duration_s"]
        .agg(["mean", "median", "max"])
        .sort_values("mean")
    )
    print("\n=== Duration (seconds) by model ===")
    print(speed.round(1).to_string())


if __name__ == "__main__":
    asyncio.run(main())
