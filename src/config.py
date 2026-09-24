"""
Central configuration: paths, column names, and split parameters.
Import this everywhere instead of hardcoding strings/paths, so the whole
pipeline stays consistent as you move from EDA -> modeling -> app.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# The Kaggle dataset (kartik2112/fraud-detection) ships as separate
# train/test CSVs already. We use fraudTest.csv as-is for final holdout
# evaluation (untouched until the very end), and carve a validation set
# out of the *tail* of fraudTrain.csv chronologically -- see data_split.py.
RAW_TRAIN_CSV = RAW_DIR / "fraudTrain.csv"
RAW_TEST_CSV = RAW_DIR / "fraudTest.csv"

PROCESSED_TRAIN_CSV = PROCESSED_DIR / "train.csv"
PROCESSED_VAL_CSV = PROCESSED_DIR / "val.csv"
PROCESSED_TEST_CSV = PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
DATETIME_COL = "trans_date_trans_time"
TARGET_COL = "is_fraud"
CARD_ID_COL = "cc_num"

CARDHOLDER_LAT_COL = "lat"
CARDHOLDER_LONG_COL = "long"
MERCHANT_LAT_COL = "merch_lat"
MERCHANT_LONG_COL = "merch_long"

AMOUNT_COL = "amt"

# ---------------------------------------------------------------------------
# Split parameters
# ---------------------------------------------------------------------------
# fraudTest.csv is kept entirely as the final holdout (untouched until final
# evaluation). We carve a validation set out of the most recent slice of
# fraudTrain.csv (time-ordered, so val is always "later" than what the model
# trained on) for use during model selection / hyperparameter tuning.
VAL_FRACTION_OF_TRAIN = 0.15

RANDOM_SEED = 42
