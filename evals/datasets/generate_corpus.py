import asyncio
import hashlib
import random
import numpy as np
import pandas as pd
from sqlalchemy import func
from conduit.async_ import ModelAsync
from conduit.config import settings
from siphon_server.database.postgres.connection import SessionLocal
from siphon_server.database.postgres.models import ProcessedContentORM


def generate_source_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


tokenizer = ModelAsync("gpt-4o")

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


async def tokenize_all(candidates: list[tuple[str, str]]) -> list[tuple[str, str, int]]:
    sem = asyncio.Semaphore(TOKENIZE_CONCURRENCY)
    counts = await asyncio.gather(*[get_token_count(sem, text) for text, _ in candidates])
    return [(text, category, count) for (text, category), count in zip(candidates, counts)]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def make_log_bins(n: int = N_BINS, lo: int = MIN_TOKENS, hi: int = MAX_TOKENS) -> list[tuple[int, int]]:
    edges = np.logspace(np.log10(lo), np.log10(hi), n + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def assign_to_bins(
    tokenized: list[tuple[str, str, int]], bins: list[tuple[int, int]]
) -> dict[int, list[tuple[str, str, int]]]:
    binned: dict[int, list[tuple[str, str, int]]] = {i: [] for i in range(len(bins))}
    for text, category, count in tokenized:
        if count == 0:
            continue
        for i, (lo, hi) in enumerate(bins):
            if lo <= count < hi:
                binned[i].append((text, category, count))
                break
    return binned


def sample_bins(
    binned: dict[int, list[tuple[str, str, int]]], bins: list[tuple[int, int]]
) -> list[tuple[str, str, int]]:
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

async def fetch_siphon_candidates() -> list[tuple[str, str]]:
    db = SessionLocal()
    try:
        # General pool: any length >= 2K chars
        short_rows = (
            db.query(ProcessedContentORM.content_text)
            .filter(func.length(ProcessedContentORM.content_text) >= 2_000)
            .order_by(func.random())
            .limit(200)
            .all()
        )
        # Long pool: bias toward high bins (approx 20K+ tokens)
        long_rows = (
            db.query(ProcessedContentORM.content_text)
            .filter(func.length(ProcessedContentORM.content_text) >= 80_000)
            .order_by(func.random())
            .limit(200)
            .all()
        )
        rows = short_rows + long_rows
        return [(r.content_text, "Siphon") for r in rows if r.content_text]
    finally:
        db.close()


def fetch_hf_texts(
    name: str,
    split: str,
    text_fn,
    category: str,
    n: int = 150,
    subset: str = None,
) -> list[tuple[str, str]]:
    from datasets import load_dataset

    label = f"{name}/{subset}" if subset else name
    print(f"   Loading {label}...")
    try:
        if subset:
            ds = load_dataset(name, subset, split=split)
        else:
            ds = load_dataset(name, split=split)
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        texts = [text_fn(item) for item in ds]
        return [(t, category) for t in texts if t and len(t.strip()) > 200]
    except Exception as e:
        print(f"   Failed ({label}): {e}")
        return []


async def build_candidate_pool() -> list[tuple[str, str]]:
    print("\nFetching candidates...")
    pool: list[tuple[str, str]] = []

    siphon = await fetch_siphon_candidates()
    print(f"   Siphon: {len(siphon)}")
    pool.extend(siphon)

    # Public HuggingFace sources, roughly ordered short -> long
    hf_sources = [
        # Short (500-5K tokens)
        dict(name="gursi26/wikihow-cleaned", split="train", n=150, category="WikiHow",
             text_fn=lambda x: x.get("text") or x.get("article") or ""),
        dict(name="billsum", split="test", n=150, category="BillSum",
             text_fn=lambda x: x["text"]),
        # Medium (5K-30K tokens)
        dict(name="ccdv/govreport-summarization", split="test", n=150, category="GovReport",
             text_fn=lambda x: x["report"]),
        dict(name="ccdv/arxiv-summarization", split="test", n=150, category="ArXiv",
             text_fn=lambda x: x["article"]),
        # Medium (5K-20K tokens) - biomedical papers
        dict(name="ccdv/pubmed-summarization", split="test", n=150, category="PubMed",
             text_fn=lambda x: x["article"]),
        # Medium-long (20K-100K tokens) - book chapters
        dict(name="kmfoda/booksum", split="test", n=150, category="BookSum",
             text_fn=lambda x: x.get("chapter") or x.get("text") or ""),
        # Long (40K-200K tokens) - more book chapters, larger pool
        dict(name="kmfoda/booksum", split="train", n=500, category="BookSum",
             text_fn=lambda x: x.get("chapter") or x.get("text") or ""),
    ]

    for src in hf_sources:
        texts = fetch_hf_texts(**src)
        print(f"   {src['name']} ({src['category']}): {len(texts)}")
        pool.extend(texts)

    # Deduplicate by leading 300 chars
    seen: dict[str, tuple[str, str]] = {}
    for text, category in pool:
        key = text[:300]
        if key not in seen:
            seen[key] = (text, category)
    pool = list(seen.values())

    print(f"\nTotal unique candidates: {len(pool)}")
    return pool


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_dataset(docs: list[tuple[str, str, int]], name: str = "summarization_corpus.parquet") -> None:
    path = f"{DATASETS_DIR}/{name}"
    records = [
        {
            "source_id": generate_source_id(text),
            "content": text,
            "category": category,
            "token_count": token_count,
        }
        for text, category, token_count in docs
    ]
    pd.DataFrame(records).to_parquet(path, index=False)
    print(f"\nSaved {len(records)} documents -> {path}")


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
