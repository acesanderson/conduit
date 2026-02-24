from __future__ import annotations
from conduit.config import settings
from conduit.core.eval.models import Document, GoldDatum, GoldSummary
from typing import TYPE_CHECKING
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardDatum,
    GoldStandardSummaryWithMetadata,
    GoldStandardEntry,
)
import pandas as pd


if TYPE_CHECKING:
    from pathlib import Path

DATASETS_DIR = settings.paths["DATASETS_DIR"]
SUMMARIZATION_DATASET_PATH = DATASETS_DIR / "summarization_corpus.parquet"
GOLD_STANDARD_DATASET_PATH = (
    settings.paths["DATASETS_DIR"] / "gold_standard_dataset.parquet"
)


def load_corpus(path: Path = SUMMARIZATION_DATASET_PATH) -> list[Document]:
    """
    Loads the corpus from disk as a list of Document objects.
    """
    df = pd.read_parquet(str(path))
    records = df.to_dict(orient="records")
    return [Document.model_validate(r) for r in records]
    # return [
    #     Document(
    #         content=row["content"],
    #         metadata={
    #             "source_id": row["source_id"],
    #             "category": row["category"],
    #             "token_count": row["token_count"],
    #         },
    #     )
    #     for _, row in df.iterrows()
    # ]
    #


def load_golden_dataset(
    path: Path = GOLD_STANDARD_DATASET_PATH,
) -> list[GoldDatum]:
    """
    Loads the golden dataset from disk as a list of GoldDatum objects.
    Parquet data is stored in a columnar format, so we need to read the entire DataFrame and then convert each row to a GoldDatum.
    """
    df = pd.read_parquet(str(path))
    records = df.to_dict(orient="records")
    return [GoldDatum.model_validate(r) for r in records]


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents")
    if docs:
        print(f"Sample: {docs[0].metadata}")
    # gold = load_golden_dataset(k)
