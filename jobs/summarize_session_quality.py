"""
summarize_session_quality — session summary quality check (Cronicle job).

Fetches N recent Claude Code sessions from claude-history DB, generates summaries
with gpt-oss (candidate) and gemini3 (gold standard), scores via LLM-as-judge,
writes results to a JSON file, and prints a score table to stdout (Cronicle live log).

Usage:
    uv run python jobs/summarize_session_quality.py               # default: 10 sessions
    uv run python jobs/summarize_session_quality.py --cron        # health-gate (for Cronicle)
    uv run python jobs/summarize_session_quality.py --dry-run     # print sessions, exit
    uv run python jobs/summarize_session_quality.py --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — anchored to project root (two levels up from this file)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
EVALS_DIR    = PROJECT_ROOT / "evals"
LOG_PATH     = PROJECT_ROOT / "jobs" / "summarize_session_quality.log"
STATUS_PATH  = PROJECT_ROOT / "jobs" / "summarize_session_quality_status.json"
RESULTS_PATH = PROJECT_ROOT / "jobs" / "summarize_session_quality_results.json"

sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)
_shutdown = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),   # captured by Cronicle live log
            logging.FileHandler(LOG_PATH),
        ],
    )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def handle_sigterm(signum, frame) -> None:
    global _shutdown
    _shutdown = True
    logger.info("SIGTERM received — will stop after current unit of work")


# ---------------------------------------------------------------------------
# Health check (--cron gate)
# ---------------------------------------------------------------------------

async def health_check() -> bool:
    try:
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        async with HeadwaterAsyncClient(host_alias="bywater") as client:
            if not await client.ping():
                logger.warning("health_check: bywater ping returned False")
                return False
        logger.info("health_check: bywater OK")
        return True
    except Exception as exc:
        logger.warning("health_check: bywater unreachable: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def write_status(status: str, **kwargs) -> None:
    STATUS_PATH.write_text(json.dumps({
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }, indent=2, default=str))


# ---------------------------------------------------------------------------
# Core run (delegates to evals/run_session_summary.py logic)
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    from run_session_summary import (
        _open_conn,
        _fetch_recent_sessions,
        _fetch_turns,
        _build_transcript,
        SessionSummarizer,
        _generate_references,
        _generate_candidates,
        _print_results,
        _GEMINI_CONFIG,
        _GPT_CONFIG,
    )
    from evals import RunInput, evaluate
    from scorer import make_gemini_judge

    logger.info("opening claude-history DB")
    with _open_conn() as conn:
        sessions = _fetch_recent_sessions(conn, args.limit)
        if not sessions:
            logger.info("no sessions found — exiting")
            return

        logger.info("fetched %d sessions, building transcripts", len(sessions))
        transcripts: dict[str, str] = {}
        for s in sessions:
            if _shutdown:
                logger.info("shutdown flag set — stopping before transcript build")
                return
            turns = _fetch_turns(conn, s["session_id"])
            transcripts[s["session_id"]] = _build_transcript(turns)

    inputs = [
        RunInput(
            source_id=s["session_id"],
            data=transcripts[s["session_id"]],
            metadata={"title": s.get("title"), "turn_count": s.get("turn_count")},
        )
        for s in sessions
        if transcripts.get(s["session_id"])
    ]
    logger.info("%d inputs with non-empty transcripts", len(inputs))

    if args.dry_run:
        for i in inputs:
            logger.info("  session=%s turns=%s title=%s",
                        i.source_id[:8],
                        (i.metadata or {}).get("turn_count"),
                        (i.metadata or {}).get("title", "")[:60])
        return

    strategy = SessionSummarizer()

    if _shutdown:
        return

    logger.info("pass 1: generating gemini3 reference summaries")
    refs = await _generate_references(inputs, strategy, _GEMINI_CONFIG)

    if _shutdown:
        return

    logger.info("pass 2: generating gpt-oss candidate summaries")
    candidates = await _generate_candidates(inputs, strategy, _GPT_CONFIG)

    if _shutdown:
        return

    logger.info("scoring %d candidates", len(candidates))
    judge = make_gemini_judge(refs)
    scores = await evaluate(candidates, judge)

    _print_results(sessions, refs, candidates, scores)

    score_values = [er.score for er in scores]
    mean_score = sum(score_values) / len(score_values) if score_values else 0.0

    results_payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_sessions": len(inputs),
        "mean_score": round(mean_score, 4),
        "sessions": [
            {
                "session_id": er.run_result.source_id,
                "score": er.score,
                "candidate_summary": er.run_result.output.output,
                "reference_summary": refs.get(er.run_result.source_id, ""),
            }
            for er in scores
        ],
    }
    RESULTS_PATH.write_text(json.dumps(results_payload, indent=2))
    logger.info("results written to %s (mean_score=%.3f)", RESULTS_PATH, mean_score)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cron",    action="store_true", help="Health-gate before running (Cronicle mode)")
    p.add_argument("--dry-run", action="store_true", help="Print sessions and exit")
    p.add_argument("--limit",   type=int, default=10, help="Number of sessions to process")
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    setup_logging()
    signal.signal(signal.SIGTERM, handle_sigterm)

    if args.dry_run:
        await run(args)
        return

    if args.cron and not await health_check():
        logger.info("health check failed — skipping run (exit 0)")
        write_status("skipped", reason="health_check_failed")
        sys.exit(0)

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        await run(args)
        write_status("success", started_at=started_at)
    except Exception:
        logger.exception("unhandled exception")
        write_status("failure", started_at=started_at)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(async_main())
