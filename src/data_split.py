"""
Use Kaggle's fraudTrain.csv / fraudTest.csv as the given train/test split,
and carve a validation set out of the *tail* of fraudTrain.csv chronologically.

    [------- fraudTrain.csv -------]   [------- fraudTest.csv -------]
    [--- train (85%) ---][val (15%)]   [------- test (untouched) ----]
    oldest                  |          (whatever date range Kaggle gave it)
                       time cutoff

fraudTest.csv is never touched until final model evaluation -- it's your
honest measure of real-world performance. The validation set (from the tail
of fraudTrain.csv) is what you use for hyperparameter tuning and model
selection along the way, since touching test repeatedly for that would
leak information about it into your choices.

This script also prints the actual date ranges of both files, since
Kaggle's docs don't spell out whether the two files overlap in time -- you
should look at the printed ranges yourself before assuming anything.

Usage:
    python src/data_split.py
"""

import pandas as pd

from config import (
    DATETIME_COL,
    PROCESSED_DIR,
    PROCESSED_TEST_CSV,
    PROCESSED_TRAIN_CSV,
    PROCESSED_VAL_CSV,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
    TARGET_COL,
    VAL_FRACTION_OF_TRAIN,
)


def load_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Make sure fraudTrain.csv and fraudTest.csv "
            f"are in {path.parent}/"
        )
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    return df


def report(name: str, split_df: pd.DataFrame):
    n = len(split_df)
    fraud_rate = split_df[TARGET_COL].mean()
    date_min = split_df[DATETIME_COL].min()
    date_max = split_df[DATETIME_COL].max()
    print(
        f"{name:>5}: {n:>9,} rows | fraud rate {fraud_rate:.4%} | "
        f"{date_min.date()} -> {date_max.date()}"
    )


def main():
    print("Loading raw CSVs...")
    train_raw = load_csv(RAW_TRAIN_CSV).sort_values(DATETIME_COL).reset_index(drop=True)
    test_df = load_csv(RAW_TEST_CSV).sort_values(DATETIME_COL).reset_index(drop=True)

    # Carve validation from the tail of fraudTrain.csv (time-ordered).
    n = len(train_raw)
    cutoff = int(n * (1 - VAL_FRACTION_OF_TRAIN))
    train_df = train_raw.iloc[:cutoff]
    val_df = train_raw.iloc[cutoff:]

    print("\nSplit summary:")
    report("train", train_df)
    report("val", val_df)
    report("test", test_df)

    # This is a check, not an assumption: tells you plainly whether
    # fraudTrain and fraudTest actually overlap in time. If they do,
    # note it in your write-up -- it affects how you interpret test
    # performance as a "future" holdout.
    train_max = train_raw[DATETIME_COL].max()
    test_min = test_df[DATETIME_COL].min()
    if test_min < train_max:
        print(
            f"\nNOTE: fraudTest.csv's earliest date ({test_min.date()}) is "
            f"BEFORE fraudTrain.csv's latest date ({train_max.date()}). "
            "The two files overlap in time -- worth mentioning as a "
            "limitation in your report rather than assuming a clean "
            "chronological cutoff."
        )
    else:
        print(
            f"\nfraudTest.csv starts ({test_min.date()}) after fraudTrain.csv "
            f"ends ({train_max.date()}) -- clean chronological holdout."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_CSV, index=False)
    val_df.to_csv(PROCESSED_VAL_CSV, index=False)
    test_df.to_csv(PROCESSED_TEST_CSV, index=False)

    print(f"\nSaved to {PROCESSED_DIR}/ as train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
