"""
Helpers for loading the processed train/val/test splits, both raw (for EDA)
and fully engineered (for modeling), consistently across scripts.
"""

import pandas as pd

from config import (
    DATETIME_COL,
    PROCESSED_DIR,
    PROCESSED_TEST_CSV,
    PROCESSED_TRAIN_CSV,
    PROCESSED_VAL_CSV,
)

FEATURES_TRAIN_CSV = PROCESSED_DIR / "train_features.csv"
FEATURES_VAL_CSV = PROCESSED_DIR / "val_features.csv"
FEATURES_TEST_CSV = PROCESSED_DIR / "test_features.csv"


def load_split(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[DATETIME_COL])


def load_train() -> pd.DataFrame:
    return load_split(PROCESSED_TRAIN_CSV)


def load_val() -> pd.DataFrame:
    return load_split(PROCESSED_VAL_CSV)


def load_test() -> pd.DataFrame:
    return load_split(PROCESSED_TEST_CSV)


def load_all_splits():
    """Return (train_df, val_df, test_df) -- raw columns only."""
    return load_train(), load_val(), load_test()


def load_combined_sorted() -> pd.DataFrame:
    """
    Load train+val+test concatenated and sorted by timestamp, with a
    `split` column tagging origin. Used for time-aware feature engineering
    (velocity, spending deviation) that legitimately needs a card's full
    transaction history, not just history from within one split.
    """
    train_df, val_df, test_df = load_all_splits()
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)
    return combined


def _require_engineered_files():
    missing = [
        p for p in (FEATURES_TRAIN_CSV, FEATURES_VAL_CSV, FEATURES_TEST_CSV)
        if not p.exists()
    ]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing engineered feature file(s): {names}\n"
            "Run `python src/build_features.py` first to generate them."
        )


def load_engineered_train() -> pd.DataFrame:
    _require_engineered_files()
    return load_split(FEATURES_TRAIN_CSV)


def load_engineered_val() -> pd.DataFrame:
    _require_engineered_files()
    return load_split(FEATURES_VAL_CSV)


def load_engineered_test() -> pd.DataFrame:
    _require_engineered_files()
    return load_split(FEATURES_TEST_CSV)
