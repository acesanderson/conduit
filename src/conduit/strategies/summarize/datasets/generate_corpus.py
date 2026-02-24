import asyncio
import pandas as pd
from sqlalchemy import func
from conduit.async_ import ModelAsync
from conduit.core.eval.models import Document
from siphon_server.database.postgres.connection import SessionLocal
from siphon_server.database.postgres.models import ProcessedContentORM
from uuid import uuid4
from conduit.config import settings


# Initialize Tokenizer (Global) # <--- As requested
# We use 'gpt-4o' for accurate counting, matching your standard
tokenizer = ModelAsync("gpt-4o")
DATASETS_DIR = settings.paths["DATASETS_DIR"]


async def get_token_count(text: str) -> int:
    """Async wrapper for ModelAsync tokenization."""
    if not text:
        return 0
    try:
        # Await the tokenization as per your snippet
        tokens = await tokenizer.tokenize(text)
        return tokens
    except Exception as e:
        # print(f"Tokenization failed: {e}")
        return 0


async def fetch_siphon_stratified() -> list[Document]:
    """
    Pulls stratified samples from Siphon DB (Async Tokenization).
    """
    db = SessionLocal()

    tiers = [
        {
            "label": "Tier_A_Tech",
            "sources": ["Article", "Doc"],
            "min_char": 7_000,
            "max_char": 45_000,
            "min_tokens": 2_000,
            "max_tokens": 10_000,
            "target": 5,
        },
        {
            "label": "Tier_B_Messy",
            "sources": ["YouTube"],
            "min_char": 45_000,
            "max_char": 160_000,
            "min_tokens": 12_000,
            "max_tokens": 40_000,
            "target": 10,
        },
        {
            "label": "Tier_C_Long",
            "sources": ["Doc", "Article", "YouTube"],
            "min_char": 160_000,
            "max_char": 2_000_000,
            "min_tokens": 40_000,
            "max_tokens": 1_000_000,
            "target": 5,
        },
    ]

    selected_docs: list[Document] = []
    print(f"Connecting to Siphon DB...")

    try:
        for tier in tiers:
            print(f"   Searching for {tier['label']} ({tier['sources']})...")

            # Synchronous SQL query (fast enough)
            query = (
                db.query(ProcessedContentORM)
                .filter(ProcessedContentORM.source_type.in_(tier["sources"]))
                .filter(
                    func.length(ProcessedContentORM.content_text) >= tier["min_char"]
                )
                .filter(
                    func.length(ProcessedContentORM.content_text) <= tier["max_char"]
                )
                .order_by(func.random())
                .limit(tier["target"] * 4)
            )

            candidates = query.all()
            found_in_tier = 0

            for orm in candidates:
                if found_in_tier >= tier["target"]:
                    break

                count = await get_token_count(orm.content_text)

                if tier["min_tokens"] <= count <= tier["max_tokens"]:
                    selected_docs.append(
                        Document(
                            content=orm.content_text,
                            metadata={
                                "source_id": orm.uri,
                                "category": tier["label"],
                                "token_count": count,
                            },
                        )
                    )
                    found_in_tier += 1

            print(f"Selected {found_in_tier}/{tier['target']} docs.")

    finally:
        db.close()

    return selected_docs


async def build_public_dataset(name, split, category, source_id_fn, text_fn) -> list[Document]:
    """
    Helper to load and tokenize public datasets asynchronously.
    """
    from datasets import load_dataset

    print(f"   Loading {name}...")
    ds = load_dataset(name, split=split).shuffle(seed=42).select(range(10))

    docs: list[Document] = []
    for i, item in enumerate(ds):
        text = text_fn(item)
        count = await get_token_count(text)
        docs.append(
            Document(
                content=text,
                metadata={
                    "category": category,
                    "source_id": source_id_fn(item, i),
                    "token_count": count,
                },
            )
        )
    return docs


async def build_composite_dataset() -> list[Document]:
    print("\nBuilding Composite Dataset (Async)...")

    # 1. Siphon Data (Async)
    siphon_docs = await fetch_siphon_stratified()

    # 2. Public Datasets (Async Iteration)

    # GovReport
    gov_docs = await build_public_dataset(
        name="ccdv/govreport-summarization",
        split="test",
        category="GovReport",
        source_id_fn=lambda x, i: str(uuid4().hex)[:8],
        text_fn=lambda x: x["report"],
    )

    # BillSum
    bill_docs = await build_public_dataset(
        name="billsum",
        split="test",
        category="BillSum",
        source_id_fn=lambda x, i: x["title"][:50],
        text_fn=lambda x: x["text"],
    )

    # WikiHow (gursi26)
    wiki_docs = await build_public_dataset(
        name="gursi26/wikihow-cleaned",
        split="train",
        category="WikiHow",
        source_id_fn=lambda x, i: f"wiki_{i}",
        text_fn=lambda x: x.get("text") or x.get("article") or x.get("input", ""),
    )

    return siphon_docs + gov_docs + bill_docs + wiki_docs


def save_dataset(docs: list[Document], name: str = "summarization_corpus.parquet") -> None:
    path = f"{DATASETS_DIR}/{name}"
    records = [{"content": d.content, **d.metadata} for d in docs]
    pd.DataFrame(records).to_parquet(path, index=False)
    print(f"Dataset saved to: {path}")


if __name__ == "__main__":
    # The Entry Point
    docs = asyncio.run(build_composite_dataset())

    print(f"\nFinal Count: {len(docs)} documents")
    if len(docs) > 0:
        from collections import Counter

        cats = Counter(d.metadata["category"] for d in docs)
        print(f"   Breakdown: {dict(cats)}")

        avg_tokens = sum(d.metadata["token_count"] for d in docs) / len(docs)
        print(f"   Avg Tokens: {int(avg_tokens)}")
        print(f"   Sample ID: {docs[0].metadata['source_id']}")

    save_dataset(docs)
