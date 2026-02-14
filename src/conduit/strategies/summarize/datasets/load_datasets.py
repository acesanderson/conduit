from __future__ import annotations
from conduit.config import settings
from datasets import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

DATASETS_DIR = settings.paths["DATASETS_DIR"]
SUMMARIZATION_DATASET_PATH = DATASETS_DIR / "summarization_corpus.parquet"
GOLD_STANDARD_DATASET_PATH = (
    settings.paths["DATASETS_DIR"] / "gold_standard_dataset.parquet"
)


def load_corpus(path: Path = SUMMARIZATION_DATASET_PATH) -> Dataset:
    """
    Loads the corpus from disk.
    """
    dataset = Dataset.from_parquet(str(path))
    return dataset


def load_golden_dataset(
    path: Path = GOLD_STANDARD_DATASET_PATH,
) -> Dataset:
    """
    Loads the golden dataset from disk.
    """
    dataset = Dataset.from_parquet(str(path))
    return dataset


if __name__ == "__main__":
    ds = load_corpus()
    print(f"Loaded dataset with {len(ds)} records.")
    ds_gold = load_golden_dataset()
    print(f"Loaded gold standard dataset with {len(ds_gold)} records.")
