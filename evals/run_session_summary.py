"""
Session summary quality check.

Fetches N recent Claude Code sessions from claude-history DB, builds transcripts,
generates summaries with gpt-oss (candidate) and gemini3 (gold standard), then
scores candidates via LLM-as-judge (gemini3).

Usage:
    python evals/run_session_summary.py               # default: 10 sessions
    python evals/run_session_summary.py --limit 5
    python evals/run_session_summary.py --dry-run     # print sessions, exit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import tiktoken
from jinja2 import Template

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evals import RunInput, RunOutput, RunResult, evaluate
from scorer import make_gemini_judge

LOG_PATH = Path(__file__).parent / "run_session_summary.log"

logger = logging.getLogger(__name__)

_BUDGET = 32_000
_HEAD_N = 40
_TAIL_N = 20

_SUMMARY_PROMPT = Template(
    "You are given a Claude Code session transcript.\n"
    "Summarize what was worked on, including concrete technical details: "
    "tools used, files changed, commands run, outcomes, and any key decisions or learnings.\n"
    "Be specific — name actual files, functions, libraries, or concepts where relevant.\n"
    "3–6 bullet points. No preamble.\n\n"
    "<transcript>\n"
    "{{ transcript }}\n"
    "</transcript>"
)


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def _open_conn(db_name: str = "claude_history"):
    from dbclients.clients.postgres import get_postgres_client
    return get_postgres_client(client_type="context_db", dbname=db_name)()


def _fetch_recent_sessions(conn, limit: int) -> list[dict]:
    import psycopg2.extras
    sql = """
    SELECT s.session_id, s.project_name, s.title,
           s.started_at, s.ended_at,
           COUNT(t.id) AS turn_count
    FROM cc_sessions s
    LEFT JOIN cc_turns t USING (session_id)
    WHERE NOT s.is_subagent
    GROUP BY s.session_id
    HAVING COUNT(t.id) > 0
    ORDER BY s.started_at DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return [dict(r) for r in cur.fetchall()]


def _fetch_turns(conn, session_id: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Transcript builder
# ---------------------------------------------------------------------------

def _build_transcript(turns: list[dict], budget: int = _BUDGET) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    nonempty = [t for t in turns if t["content_text"]]
    if not nonempty:
        return ""

    head = nonempty[:_HEAD_N]
    tail = [t for t in nonempty[-_TAIL_N:] if t not in head]
    selected = head + tail

    per_turn_budget = budget // len(selected)

    parts = []
    for t in selected:
        tokens = enc.encode(t["content_text"])
        text = enc.decode(tokens[:per_turn_budget]) if len(tokens) > per_turn_budget else t["content_text"]
        parts.append(f"{t['role'].upper()}: {text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class SessionSummarizer:
    """Summarizes a session transcript. Reads config['model'] and config.get('use_remote')."""

    async def __call__(self, input: RunInput, config: dict) -> str:
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.utils.progress.verbosity import Verbosity

        model_name = config["model"]
        prompt = _SUMMARY_PROMPT.render(transcript=input.data)
        params = GenerationParams(model=model_name, temperature=0.0)
        options = ConduitOptions(
            project_name="session_summary_eval",
            verbosity=Verbosity.SILENT,
            include_history=False,
        )

        if config.get("use_remote"):
            from conduit.remote import RemoteModelAsync
            model = RemoteModelAsync(model=model_name, host_alias=config.get("host_alias", "headwater"))
        else:
            from conduit.core.model.model_async import ModelAsync
            model = ModelAsync(model=model_name)

        response = await model.query(query_input=prompt, params=params, options=options)
        return str(response.content).strip()


# ---------------------------------------------------------------------------
# Two-pass eval
# ---------------------------------------------------------------------------

async def _generate_references(
    inputs: list[RunInput],
    strategy: SessionSummarizer,
    config: dict,
) -> dict[str, str]:
    """Pass 1: run gemini3 on all inputs, return source_id → summary."""
    sem = asyncio.Semaphore(5)

    async def _one(inp: RunInput) -> tuple[str, str]:
        async with sem:
            summary = await strategy(inp, config)
            logger.info("ref generated  session=%s  chars=%d", inp.source_id[:8], len(summary))
            return inp.source_id, summary

    pairs = await asyncio.gather(*[_one(i) for i in inputs])
    return dict(pairs)


async def _generate_candidates(
    inputs: list[RunInput],
    strategy: SessionSummarizer,
    config: dict,
) -> list[RunResult]:
    """Pass 2: run gpt-oss on all inputs, return RunResult list."""
    import hashlib, json
    sem = asyncio.Semaphore(5)
    config_id = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

    async def _one(inp: RunInput) -> RunResult | None:
        async with sem:
            try:
                summary = await strategy(inp, config)
                logger.info("candidate done session=%s  chars=%d", inp.source_id[:8], len(summary))
                return RunResult(
                    strategy=strategy.__class__.__name__,
                    config_id=config_id,
                    source_id=inp.source_id,
                    config=config,
                    output=RunOutput(output=summary, metadata={}),
                )
            except Exception as exc:
                logger.error("candidate failed session=%s: %s: %s",
                             inp.source_id[:8], type(exc).__name__, exc)
                return None

    results = await asyncio.gather(*[_one(i) for i in inputs])
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results(
    sessions: list[dict],
    refs: dict[str, str],
    candidates: list[RunResult],
    scores: list,
) -> None:
    score_map = {er.run_result.source_id: er.score for er in scores}
    meta_map  = {s["session_id"]: s for s in sessions}

    header = f"{'session_id':<38} {'started':<12} {'turns':>5} {'chars':>6} {'score':>6}  title"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in candidates:
        meta  = meta_map.get(r.source_id, {})
        score = score_map.get(r.source_id, 0.0)
        started = str(meta.get("started_at", ""))[:10]
        turns   = meta.get("turn_count", "?")
        chars   = len(r.output.output)
        title   = (meta.get("title") or "")[:50]
        print(f"{r.source_id:<38} {started:<12} {turns:>5} {chars:>6} {score:>6.3f}  {title}")

    valid_scores = [er.score for er in scores]
    if valid_scores:
        print("-" * len(header))
        print(f"{'mean':>60}  {sum(valid_scores)/len(valid_scores):.3f}")
        print(f"{'min':>60}  {min(valid_scores):.3f}")
        print(f"{'max':>60}  {max(valid_scores):.3f}")

    print("\n--- Sample: first candidate ---")
    if candidates:
        sid = candidates[0].source_id
        print(f"\n[gpt-oss]\n{candidates[0].output.output}")
        if sid in refs:
            print(f"\n[gemini3 reference]\n{refs[sid]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_GEMINI_CONFIG = {"model": "gemini3"}
_GPT_CONFIG    = {"model": "gpt-oss:latest", "use_remote": True, "host_alias": "bywater"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH),
        ],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    setup_logging()

    with _open_conn() as conn:
        sessions = _fetch_recent_sessions(conn, args.limit)
        if not sessions:
            print("No sessions found.")
            return

        print(f"Fetched {len(sessions)} sessions.")

        transcripts: dict[str, str] = {}
        for s in sessions:
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

    if args.dry_run:
        for i in inputs:
            meta = i.metadata or {}
            toks = len(i.data.split()) * 4 // 3  # rough estimate
            print(f"  {i.source_id}  turns={meta.get('turn_count')}  ~{toks} tokens  {meta.get('title', '')[:60]}")
        return

    strategy = SessionSummarizer()

    print(f"\nPass 1: generating gemini3 reference summaries ({len(inputs)} sessions)...")
    refs = await _generate_references(inputs, strategy, _GEMINI_CONFIG)

    print(f"\nPass 2: generating gpt-oss candidate summaries ({len(inputs)} sessions)...")
    candidates = await _generate_candidates(inputs, strategy, _GPT_CONFIG)

    MIN_CHARS = 50
    valid_candidates = [c for c in candidates if len(c.output.output) >= MIN_CHARS]
    dropped = len(candidates) - len(valid_candidates)
    if dropped:
        print(f"  dropped {dropped} near-empty candidates (< {MIN_CHARS} chars)")
    candidates = valid_candidates

    print("\nScoring candidates against references...")
    judge = make_gemini_judge(refs)
    scores = await evaluate(candidates, judge)

    _print_results(sessions, refs, candidates, scores)


if __name__ == "__main__":
    asyncio.run(main())
