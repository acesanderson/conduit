from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from conduit.config import settings
from conduit.core.eval.models import GoldDatum, GoldSummary
from evals import RunInput
from typing import TYPE_CHECKING
from gold_standard import (
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


def load_corpus(path: Path = SUMMARIZATION_DATASET_PATH) -> list[RunInput]:
    """
    Loads the corpus from disk as a list of RunInput objects.
    """
    df = pd.read_parquet(str(path))
    records = df.to_dict(orient="records")
    return [
        RunInput(
            source_id=r["source_id"],
            data=r["content"],
            metadata={"token_count": r["token_count"], "category": r["category"]},
        )
        for r in records
    ]


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
        print(f"Sample source_id: {docs[0].source_id}")
        print(f"Sample metadata: {docs[0].metadata}")
