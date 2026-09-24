"""
Builds per-card transaction sequences for the LSTM (Milestone 4b).

Every other model so far treats each transaction as an independent row.
The LSTM instead sees, for each transaction, the preceding SEQ_LEN-1
transactions on the same card plus the current one -- letting it learn
patterns in the *sequence* itself (e.g. a slow ramp-up in amount, a sudden
burst) rather than only summary statistics of it.

Built via lagged columns (`groupby(card).shift(lag)`), not a Python loop
per row -- this is the vectorized way to turn a flat time-ordered
DataFrame into per-entity sequences in pandas, and it scales to the full
~1.85M-row dataset without needing dedicated sequence-building libraries.
Cards with fewer than SEQ_LEN prior transactions get zero-padded at the
front of their sequence (a simplification worth naming as such in your
report, rather than the more involved packed-sequence/masking approach).
"""

import numpy as np
import pandas as pd

from config import CARD_ID_COL, DATETIME_COL
from preprocessing import BINARY_FEATURES, NUMERIC_FEATURES

SEQ_LEN = 5

# The per-timestep (dynamic) features the LSTM sees at each position in the
# sequence -- the same numeric/binary features used by the tabular models,
# so a timestep here means the same thing a row meant to XGBoost/the FNN.
SEQ_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# Context that describes the *current* transaction but isn't meaningfully
# "sequential" (a customer's age or home city population doesn't change
# transaction-to-transaction) -- fed to the model once, alongside the
# LSTM's output, rather than repeated at every timestep.
STATIC_NUMERIC = ["customer_age", "city_pop_log"]
STATIC_CATEGORICAL = ["category", "gender"]


def build_lagged_sequences(df: pd.DataFrame, seq_features=SEQ_FEATURES, seq_len=SEQ_LEN) -> np.ndarray:
    """
    Returns an array of shape (n_rows, seq_len, len(seq_features)), float32.
    `df` must already be sorted by DATETIME_COL ascending (globally --
    groupby preserves each group's relative row order, so a global time
    sort is sufficient for each card's rows to come out in time order).

    Position seq_len-1 (the last timestep) is the current transaction;
    position 0 is seq_len-1 transactions before it. Missing history
    (early transactions for a card) is zero-filled.
    """
    grouped = df.groupby(CARD_ID_COL, group_keys=False, sort=False)

    lag_slices = []
    for lag in range(seq_len - 1, -1, -1):
        if lag == 0:
            lag_df = df[seq_features]
        else:
            lag_df = grouped[seq_features].shift(lag)
        lag_slices.append(lag_df.fillna(0.0).to_numpy(dtype=np.float32))

    # Stack along a new axis=1 -> (n_rows, seq_len, n_features)
    return np.stack(lag_slices, axis=1)
