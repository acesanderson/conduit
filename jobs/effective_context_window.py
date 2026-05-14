from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from effective_context_window import ECWEvalRunner
from load_datasets import load_golden_dataset
from scorer import make_gemini_judge
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure effective context window via summarization quality degradation"
    )
    p.add_argument("--model",     required=True, help="Model name (e.g. qwen3.6:latest)")
    p.add_argument("--server",    required=True, help="Server alias (e.g. deepwater, bywater)")
    p.add_argument("--token-cap", type=int, default=100_000, help="Max token count per doc (default: 100000)")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--cron",      action="store_true")
    p.add_argument("--limit",     type=int)
    p.add_argument("--project",   help="DB project name (default: ecw_{model_slug})")
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
