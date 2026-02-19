from __future__ import annotations
from conduit.config import settings
from datasets import Dataset
from typing import TYPE_CHECKING
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardDatum,
    GoldStandardSummaryWithMetadata,
    GoldStandardEntry,
)


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


def load_datums() -> list[GoldStandardDatum]:
    """
    Loads the golden dataset and converts it to a list of GoldStandardDatum objects.
    """
    ds = load_golden_dataset()

    datums = []

    # Iterate through the columns using zip to reconstruct the objects
    for entry_data, summary_data in zip(ds["entry"], ds["summary"]):
        # entry_data is a dict: {'category', 'source_id', 'text', 'token_count'}
        # summary_data is a dict: {'entity_list', 'entity_list_embeddings', ...}

        datum = GoldStandardDatum(
            entry=GoldStandardEntry(**entry_data),
            summary=GoldStandardSummaryWithMetadata(**summary_data),
        )
        datums.append(datum)

    return datums


if __name__ == "__main__":
    datums = load_datums()
    # ds = load_golden_dataset()
