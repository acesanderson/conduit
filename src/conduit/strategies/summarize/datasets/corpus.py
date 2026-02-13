import asyncio
import sys
from datasets import Dataset, load_dataset, concatenate_datasets, Features, Value
from sqlalchemy import func
from conduit.async_ import ModelAsync  # <--- As requested
from siphon_server.database.postgres.connection import SessionLocal
from siphon_server.database.postgres.models import ProcessedContentORM
from uuid import uuid4

# Initialize Tokenizer (Global)
# We use 'gpt-4o' for accurate counting, matching your standard
tokenizer = ModelAsync("gpt-4o")


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


async def fetch_siphon_stratified():
    """
    Pulls stratified samples from Siphon DB (Async Tokenization).
    """
    db = SessionLocal()

    # Explicit Schema to prevent crashes on empty selections
    features = Features(
        {
            "source_id": Value("string"),
            "text": Value("string"),
            "category": Value("string"),
            "token_count": Value("int64"),
        }
    )

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

    selected_docs = []
    print(f"🔌 Connecting to Siphon DB...")

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

                # AWAIT the tokenization here
                count = await get_token_count(orm.content_text)

                if tier["min_tokens"] <= count <= tier["max_tokens"]:
                    selected_docs.append(
                        {
                            "source_id": orm.uri,
                            "text": orm.content_text,
                            "category": tier["label"],
                            "token_count": count,
                        }
                    )
                    found_in_tier += 1

            print(f"     ✅ Selected {found_in_tier}/{tier['target']} docs.")

    finally:
        db.close()

    # Return safe empty dataset if nothing found
    if not selected_docs:
        return Dataset.from_dict(
            {"source_id": [], "text": [], "category": [], "token_count": []},
            features=features,
        )

    return Dataset.from_list(selected_docs, features=features)


async def build_public_dataset(name, split, category, source_id_fn, text_fn):
    """
    Helper to load and tokenize public datasets asynchronously without using .map()
    """
    print(f"   Loading {name}...")
    ds = load_dataset(name, split=split).shuffle(seed=42).select(range(10))

    processed = []
    for i, item in enumerate(ds):
        text = text_fn(item)
        count = await get_token_count(text)

        processed.append(
            {
                "category": category,
                "source_id": source_id_fn(item, i),
                "text": text,
                "token_count": count,
            }
        )
    return Dataset.from_list(processed)


async def build_composite_dataset():
    print("\n🏗️  Building Composite Dataset (Async)...")

    # 1. Siphon Data (Async)
    siphon_ds = await fetch_siphon_stratified()

    # 2. Public Datasets (Async Iteration)

    # GovReport
    gov_ds = await build_public_dataset(
        name="ccdv/govreport-summarization",
        split="test",
        category="GovReport",
        source_id_fn=lambda x, i: str(uuid4().hex)[:8],
        text_fn=lambda x: x["report"],
    )

    # BillSum
    bill_ds = await build_public_dataset(
        name="billsum",
        split="test",
        category="BillSum",
        source_id_fn=lambda x, i: x["title"][:50],
        text_fn=lambda x: x["text"],
    )

    # WikiHow (gursi26)
    wiki_ds = await build_public_dataset(
        name="gursi26/wikihow-cleaned",
        split="train",
        category="WikiHow",
        source_id_fn=lambda x, i: f"wiki_{i}",
        text_fn=lambda x: x.get("text") or x.get("article") or x.get("input", ""),
    )

    # 3. Merge
    columns = ["source_id", "text", "category", "token_count"]

    combined = concatenate_datasets(
        [
            siphon_ds.select_columns(columns),
            gov_ds.select_columns(columns),
            bill_ds.select_columns(columns),
            wiki_ds.select_columns(columns),
        ]
    )

    return combined


if __name__ == "__main__":
    # The Entry Point
    ds = asyncio.run(build_composite_dataset())

    print(f"\n📦 Final Count: {len(ds)} documents")
    if len(ds) > 0:
        from collections import Counter

        cats = Counter(ds["category"])
        print(f"   Breakdown: {dict(cats)}")

        avg_tokens = sum(ds["token_count"]) / len(ds)
        print(f"   Avg Tokens: {int(avg_tokens)}")
        print(f"   Sample ID: {ds[0]['source_id']}")
