"""
backfill_summaries — generate gemini3 summaries for all unsummarized sessions (Cronicle job).

Runs the DB migration (adds summary/summarized_at columns if missing), then processes
all sessions that have no summary in batches. Uses gemini3 via conduit.

Usage:
    uv run python jobs/backfill_summaries.py               # process all unsummarized
    uv run python jobs/backfill_summaries.py --batch 50    # override batch size
    uv run python jobs/backfill_summaries.py --dry-run     # count + print, no writes
    uv run python jobs/backfill_summaries.py --cron        # health-gate (for Cronicle)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CLAUDE_HISTORY_SRC = Path.home() / "vibe" / "claude-history-project" / "src"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(CLAUDE_HISTORY_SRC))

logger = logging.getLogger(__name__)
_shutdown = False


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def handle_sigterm(signum, frame) -> None:
    global _shutdown
    _shutdown = True
    logger.info("SIGTERM received — will stop after current session")


def _open_conn():
    from dbclients.clients.postgres import get_postgres_client
    return get_postgres_client(client_type="context_db", dbname="claude_history")()


def _fetch_turns(conn, session_id: str) -> list:
    import psycopg2.extras
    sql = """
    SELECT seq, role, content_text
    FROM cc_turns
    WHERE session_id = %s
    ORDER BY seq
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (session_id,))
        return [dict(r) for r in cur.fetchall()]


def run(args: argparse.Namespace) -> None:
    from claude_history.db import migrate_add_summary, sessions_without_summaries, update_summary
    from claude_history.models import Turn
    from claude_history.summarizer import generate_summary

    logger.info("opening claude-history DB")
    conn = _open_conn()

    logger.info("running migration (add summary columns if needed)")
    migrate_add_summary(conn)

    pending = sessions_without_summaries(conn)
    logger.info("%d sessions without summaries", len(pending))

    if args.dry_run:
        logger.info("dry-run: would process %d sessions", len(pending))
        conn.close()
        return

    done = 0
    skipped = 0
    failed = 0

    for session_id in pending:
        if _shutdown:
            logger.info("shutdown flag — stopping (done=%d)", done)
            break

        try:
            turn_rows = _fetch_turns(conn, str(session_id))
            if not turn_rows:
                skipped += 1
                continue

            turns = [
                Turn(
                    session_id=str(session_id),
                    seq=r["seq"],
                    role=r["role"],
                    content_text=r["content_text"] or "",
                    content_raw={},
                    ts=None,
                )
                for r in turn_rows
            ]

            summary = generate_summary(turns)
            if not summary or len(summary) < 20:
                logger.warning("session=%s: summary too short (%d chars) — skipping",
                               str(session_id)[:8], len(summary or ""))
                skipped += 1
                continue

            update_summary(conn, str(session_id), summary)
            done += 1
            logger.info("session=%s  done  chars=%d  total=%d",
                        str(session_id)[:8], len(summary), done)

        except Exception as exc:
            logger.error("session=%s  failed: %s: %s",
                         str(session_id)[:8], type(exc).__name__, exc)
            failed += 1

    conn.close()
    logger.info("backfill complete: done=%d  skipped=%d  failed=%d", done, skipped, failed)
    if failed > 0:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch",   type=int, default=0, help="(unused, reserved for future chunking)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cron",    action="store_true", help="Health-gate: exit 0 silently if DB unreachable")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    signal.signal(signal.SIGTERM, handle_sigterm)

    if args.cron:
        try:
            conn = _open_conn()
            conn.close()
        except Exception as exc:
            logger.info("health check failed — skipping run: %s", exc)
            sys.exit(0)

    try:
        run(args)
    except Exception:
        logger.exception("unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
