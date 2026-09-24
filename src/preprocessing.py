"""
Defines the feature set that goes into the models, and builds the
scikit-learn preprocessing transformer (scaling + encoding) applied to it.

Feature choices here are backed by the EDA notebook (notebooks/01_eda.ipynb):

- `amt_zscore_vs_card_history` and `txn_count_1h` showed the strongest
  separation between fraud/legitimate of anything in the dataset.
- `distance_mi` showed almost no separation but is kept anyway -- free for
  tree models, and worth reporting either way.
- High-cardinality identifiers (cc_num, merchant, job, street, city, state,
  zip, first/last name, trans_num) are deliberately excluded from this
  baseline via the ColumnTransformer's `remainder="drop"`.
- `gender` is included because EDA showed a mild fraud-rate difference by
  gender, but it's worth flagging as a fairness consideration in the report.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import TARGET_COL

NUMERIC_FEATURES = [
    "amt",
    "amt_zscore_vs_card_history",
    "txn_count_1h",
    "txn_amt_sum_1h",
    "txn_count_24h",
    "txn_amt_sum_24h",
    "distance_mi",
    "customer_age",
    "city_pop_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

BINARY_FEATURES = [
    "is_weekend",
    "is_night",
]

CATEGORICAL_FEATURES = [
    "category",
    "gender",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def get_X_y(df):
    """Slice out the modeling feature columns and target from an engineered df."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected engineered columns: {missing}. "
            "Did you run `python src/build_features.py`?"
        )
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer: scale numeric features, one-hot encode categoricals,
    pass binary flags through unchanged. Fit this on TRAIN ONLY, then
    .transform() val/test -- never .fit() on val or test data.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """Wrap a classifier with the standard preprocessing step ahead of it."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
