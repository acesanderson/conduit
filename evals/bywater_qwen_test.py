"""
Quick timing test: qwen3.6 on bywater (RTX 4090, ~5% CPU offload).
Runs 10 docs, reports cold boot and warm boot durations.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evals import generate_runs
from load_datasets import load_golden_dataset
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer


async def main() -> None:
    docs = load_golden_dataset()[:10]
    strategy = RecursiveSummarizer()
    config = [{"model": "qwen3.6:latest", "use_remote": True, "host_alias": "bywater", "use_cache": False}]

    print(f"Running {len(docs)} docs on bywater with qwen3.6:latest ...")
    results = await generate_runs(inputs=docs, configs=config, strategy=strategy)
    print(f"Got {len(results)} results.\n")

    durations = []
    for r in results:
        trace = r.output.metadata.get("trace", [])
        d = trace[0]["duration"] if trace else None
        durations.append(d)
        cached = d is not None and d < 2.0
        flag = " [cache]" if cached else ""
        print(f"  {r.source_id[:16]}  {f'{d:.1f}s' if d else 'n/a':>8}{flag}")

    real = [d for d in durations if d is not None and d >= 2.0]
    if real:
        import statistics
        print(f"\nCold boot (1st real request): {real[0]:.1f}s")
        print(f"Warm median:                  {statistics.median(real):.1f}s")
        print(f"Warm mean:                    {statistics.mean(real):.1f}s")
        print(f"Max:                          {max(real):.1f}s")
        print(f"Real inference requests:      {len(real)}/10")
    else:
        print("\nAll requests were cache hits — no real inference measured.")
        print("Wipe the conduit cache for these docs to get real timings.")


if __name__ == "__main__":
    asyncio.run(main())
