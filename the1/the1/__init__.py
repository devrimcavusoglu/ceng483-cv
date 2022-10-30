from pathlib import Path

SRC_DIR = Path(__file__).parent.parent
DATASET_DIR = SRC_DIR / "dataset"
DATASET_CACHE_DIR = SRC_DIR / ".dataset"
QUERY_DIRS = [
    DATASET_DIR / "query_1",
    DATASET_DIR / "query_2",
    DATASET_DIR / "query_3",
]
SUPPORT_DIR = DATASET_DIR / "support_96"
CACHE_FILENAME = "support.pkl"
