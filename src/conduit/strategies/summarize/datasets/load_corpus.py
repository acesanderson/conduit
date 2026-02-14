from __future__ import annotations
from conduit.config import settings
from datasets import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

DATASETS_DIR = settings.paths["DATASETS_DIR"]
SUMMARIZATION_DATASET = DATASETS_DIR / "summarization_corpus.parquet"


def load_corpus(path: Path = SUMMARIZATION_DATASET) -> Dataset:
    """
    Loads the corpus from disk.
    """
    dataset = Dataset.from_parquet(str(path))
    return dataset


if __name__ == "__main__":
    ds = load_corpus()
    print(f"Loaded dataset with {len(ds)} records.")
