import asyncio
import random
import numpy as np
import pandas as pd
from sqlalchemy import func
from conduit.async_ import ModelAsync
from conduit.config import settings
from siphon_server.database.postgres.connection import SessionLocal
from siphon_server.database.postgres.models import ProcessedContentORM

tokenizer = ModelAsync("gpt3")
DATASETS_DIR = settings.paths["DATASETS_DIR"]

N_BINS = 20
MIN_TOKENS = 500
MAX_TOKENS = 200_000
DOCS_PER_BIN = 10
TOKENIZE_CONCURRENCY = 20


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

async def get_token_count(sem: asyncio.Semaphore, text: str) -> int:
    async with sem:
        if not text:
            return 0
        try:
            return await tokenizer.tokenize(text)
        except Exception:
            return 0


async def tokenize_all(texts: list[str]) -> list[tuple[str, int]]:
    sem = asyncio.Semaphore(TOKENIZE_CONCURRENCY)
    counts = await asyncio.gather(*[get_token_count(sem, t) for t in texts])
    return list(zip(texts, counts))


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def make_log_bins(n: int = N_BINS, lo: int = MIN_TOKENS, hi: int = MAX_TOKENS) -> list[tuple[int, int]]:
    edges = np.logspace(np.log10(lo), np.log10(hi), n + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def assign_to_bins(
    tokenized: list[tuple[str, int]], bins: list[tuple[int, int]]
) -> dict[int, list[str]]:
    binned: dict[int, list[str]] = {i: [] for i in range(len(bins))}
    for text, count in tokenized:
        if count == 0:
            continue
        for i, (lo, hi) in enumerate(bins):
            if lo <= count < hi:
                binned[i].append(text)
                break
    return binned


def sample_bins(
    binned: dict[int, list[str]], bins: list[tuple[int, int]]
) -> list[str]:
    random.seed(42)
    selected = []
    print("\nBin fill report:")
    for i, (lo, hi) in enumerate(bins):
        available = binned[i]
        n = min(DOCS_PER_BIN, len(available))
        sample = random.sample(available, n)
        selected.extend(sample)
        tag = "OK" if n == DOCS_PER_BIN else f"SPARSE ({n}/{DOCS_PER_BIN})"
        print(f"   Bin {i + 1:2d}  [{lo:>7,} – {hi:>7,} tok]  {tag}")
    return selected


# ---------------------------------------------------------------------------
# Candidate sources
# ---------------------------------------------------------------------------

async def fetch_siphon_candidates() -> list[str]:
    db = SessionLocal()
    try:
        rows = (
            db.query(ProcessedContentORM.content_text)
            .filter(func.length(ProcessedContentORM.content_text) >= 2_000)
            .order_by(func.random())
            .limit(400)
            .all()
        )
        return [r.content_text for r in rows if r.content_text]
    finally:
        db.close()


def fetch_hf_texts(
    name: str,
    split: str,
    text_fn,
    n: int = 150,
    subset: str = None,
) -> list[str]:
    from datasets import load_dataset

    label = f"{name}/{subset}" if subset else name
    print(f"   Loading {label}...")
    try:
        if subset:
            ds = load_dataset(name, subset, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        texts = [text_fn(item) for item in ds]
        return [t for t in texts if t and len(t.strip()) > 200]
    except Exception as e:
        print(f"   Failed ({label}): {e}")
        return []


async def build_candidate_pool() -> list[str]:
    print("\nFetching candidates...")
    pool: list[str] = []

    # Internal Siphon DB
    siphon = await fetch_siphon_candidates()
    print(f"   Siphon: {len(siphon)}")
    pool.extend(siphon)

    # Public HuggingFace sources, roughly ordered short → long
    hf_sources = [
        # Short (500–5K tokens)
        dict(
            name="gursi26/wikihow-cleaned", split="train", n=150,
            text_fn=lambda x: x.get("text") or x.get("article") or "",
        ),
        dict(
            name="billsum", split="test", n=150,
            text_fn=lambda x: x["text"],
        ),
        # Medium (5K–30K tokens)
        dict(
            name="ccdv/govreport-summarization", split="test", n=150,
            text_fn=lambda x: x["report"],
        ),
        dict(
            name="ccdv/arxiv-summarization", split="test", n=150,
            text_fn=lambda x: x["article"],
        ),
        # Medium-long (20K–80K tokens) — TV transcripts and meeting transcripts
        dict(
            name="tau/scrolls", subset="summ_screen_fd", split="test", n=150,
            text_fn=lambda x: x["input"],
        ),
        dict(
            name="tau/scrolls", subset="qmsum", split="test", n=150,
            text_fn=lambda x: x["input"],
        ),
        # Long (40K–200K tokens)
        dict(
            name="kmfoda/booksum", split="test", n=150,
            text_fn=lambda x: x.get("chapter") or x.get("text") or "",
        ),
        # Very long — SEC 10-K filings; tries common field names across versions
        dict(
            name="eloukas/edgar-corpus", split="train", n=150,
            text_fn=lambda x: (
                x.get("text")
                or x.get("item_1")
                or x.get("item_1a")
                or ""
            ),
        ),
    ]

    for src in hf_sources:
        texts = fetch_hf_texts(**src)
        label = src["name"] + ("/" + src["subset"] if "subset" in src else "")
        print(f"   {label}: {len(texts)}")
        pool.extend(texts)

    # Deduplicate by leading 300 chars
    seen: dict[str, str] = {}
    for t in pool:
        key = t[:300]
        if key not in seen:
            seen[key] = t
    pool = list(seen.values())

    print(f"\nTotal unique candidates: {len(pool)}")
    return pool


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_dataset(texts: list[str], name: str = "summarization_corpus.parquet") -> None:
    path = f"{DATASETS_DIR}/{name}"
    pd.DataFrame({"text": texts}).to_parquet(path, index=False)
    print(f"\nSaved {len(texts)} documents → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    bins = make_log_bins()
    pool = await build_candidate_pool()

    print(f"\nTokenizing {len(pool)} candidates...")
    tokenized = await tokenize_all(pool)

    binned = assign_to_bins(tokenized, bins)
    docs = sample_bins(binned, bins)

    print(f"\nFinal count: {len(docs)} documents")
    save_dataset(docs)


if __name__ == "__main__":
    asyncio.run(main())
