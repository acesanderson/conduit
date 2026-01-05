from conduit.config import settings
from datasets import load_dataset

DATASETS_DIR = settings.paths["DATA_DIR"] / "datasets"

# 1. Standard News (Baseline)
# '3.0.0' is the generally accepted version
# cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="test")
# load ccn_dm INTO DATASETS_DIR
cnn_dm = load_dataset(
    "cnn_dailymail", "3.0.0", split="test", cache_dir=str(DATASETS_DIR)
)
# 2. Abstractive Challenge (XSum)
# FIX: Use 'EdinburghNLP/xsum' and point to the parquet revision
xsum = load_dataset(
    "EdinburghNLP/xsum",
    split="test",
    cache_dir=str(DATASETS_DIR),
    revision="refs/convert/parquet",
)

# 3. Conversational (SAMSum)
# FIX: Use 'knkarthick/samsum' and point to the parquet revision
samsum = load_dataset(
    "knkarthick/samsum",
    split="test",
    cache_dir=str(DATASETS_DIR),
    revision="refs/convert/parquet",
)

# 4. Long-Context (SCROLLS)
# SCROLLS is complex. If the 'revision' trick fails for specific configs,
# we switch to 'tau/sled' or specific mirrors.
# Attempting the standard parquet export first:
try:
    gov_report = load_dataset(
        "tau/scrolls",
        "gov_report",
        split="test",
        cache_dir=str(DATASETS_DIR),
        revision="refs/convert/parquet",
    )

    screenplays = load_dataset(
        "tau/scrolls",
        "summ_screen_fd",
        split="test",
        cache_dir=str(DATASETS_DIR),
        revision="refs/convert/parquet",
    )
except Exception as e:
    print(f"Standard SCROLLS load failed: {e}")
    print("Falling back to direct mirrors for SCROLLS...")

    # Fallback: CCDV maintains excellent clean versions of GovReport
    gov_report = load_dataset(
        "ccdv/govreport-summarization", split="test", cache_dir=str(DATASETS_DIR)
    )
    # Screenplays might need to be skipped or found in 'tau/sled' if this fails
    print("Loaded GovReport from fallback (CCDV).")


# Verification
print(f"CNN/DM Size: {len(cnn_dm)}")
print(f"XSum Size: {len(xsum)}")
print(f"GovReport Size: {len(gov_report)}")
