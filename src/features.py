"""
Reusable feature-engineering functions, shared between the EDA notebook and
the modeling pipeline so the two never drift out of sync.

All functions are careful about one thing: a feature that looks at "this
card's past behavior" must only look BACKWARD in time relative to the
transaction being scored. Every rolling/expanding function here uses
`closed="left"` or an explicit `.shift(1)` for exactly that reason -- so
none of this leaks label information, only genuinely-available history.
"""

import numpy as np
import pandas as pd

from config import (
    AMOUNT_COL,
    CARD_ID_COL,
    CARDHOLDER_LAT_COL,
    CARDHOLDER_LONG_COL,
    DATETIME_COL,
    MERCHANT_LAT_COL,
    MERCHANT_LONG_COL,
)

EARTH_RADIUS_KM = 6371.0088
KM_TO_MILES = 0.621371


def haversine_distance_km(lat1, lon1, lat2, lon2) -> pd.Series:
    """Great-circle distance in km between two lat/long points (vectorized)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `distance_mi`: cardholder-to-merchant distance in miles.

    Note (from EDA): this feature showed almost no separation between fraud
    and legitimate transactions in this dataset (~47.3 mi mean either way).
    Kept in the pipeline anyway -- costs a tree-based model nothing -- but
    don't expect it to carry much weight.
    """
    df = df.copy()
    dist_km = haversine_distance_km(
        df[CARDHOLDER_LAT_COL],
        df[CARDHOLDER_LONG_COL],
        df[MERCHANT_LAT_COL],
        df[MERCHANT_LONG_COL],
    )
    df["distance_mi"] = dist_km * KM_TO_MILES
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the transaction timestamp."""
    df = df.copy()
    dt = pd.to_datetime(df[DATETIME_COL])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = dt.dt.month
    # EDA found fraud clusters ~10pm-3am -- flag it explicitly as a feature.
    df["is_night"] = df["hour"].between(22, 23).astype(int) | df["hour"].between(0, 3).astype(int)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings of hour-of-day and day-of-week."""
    df = df.copy()
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        df = add_datetime_features(df)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_customer_age(df: pd.DataFrame, dob_col: str = "dob") -> pd.DataFrame:
    """Add `customer_age` in years at the time of the transaction."""
    df = df.copy()
    dob = pd.to_datetime(df[dob_col])
    trans_dt = pd.to_datetime(df[DATETIME_COL])
    df["customer_age"] = (trans_dt - dob).dt.days // 365
    return df


def add_log_population(df: pd.DataFrame, pop_col: str = "city_pop") -> pd.DataFrame:
    """Add `city_pop_log`: log1p of city population (heavily right-skewed)."""
    df = df.copy()
    df["city_pop_log"] = np.log1p(df[pop_col])
    return df


def add_velocity_features(
    df: pd.DataFrame,
    windows=("1h", "24h"),
) -> pd.DataFrame:
    """
    Add per-card transaction velocity features: for each transaction, how
    many prior transactions did this same card make in the past N hours,
    and what did they total? `closed="left"` excludes the current
    transaction itself -- only strictly-past transactions count.

    `df` must already be sorted by DATETIME_COL ascending and should span
    the full timeline (train+val+test concatenated) so a card's history
    isn't artificially reset at a split boundary.
    """
    df = df.sort_values(DATETIME_COL).copy()
    df = df.set_index(pd.DatetimeIndex(df[DATETIME_COL]))

    grouped = df.groupby(CARD_ID_COL, group_keys=False)

    for w in windows:
        count_col = f"txn_count_{w}"
        sum_col = f"txn_amt_sum_{w}"
        df[count_col] = grouped[AMOUNT_COL].apply(
            lambda s: s.rolling(w, closed="left").count()
        )
        df[sum_col] = grouped[AMOUNT_COL].apply(
            lambda s: s.rolling(w, closed="left").sum()
        )
        df[count_col] = df[count_col].fillna(0)
        df[sum_col] = df[sum_col].fillna(0)

    df = df.reset_index(drop=True)
    return df


def add_spending_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `amt_zscore_vs_card_history`: how many standard deviations this
    transaction's amount is from the card's own historical (past-only)
    mean spend. Strongest signal found in EDA (fraud median z-score ~3.1
    vs ~0 for legitimate transactions).
    """
    df = df.sort_values(DATETIME_COL).copy()
    grouped = df.groupby(CARD_ID_COL)[AMOUNT_COL]

    running_mean = grouped.apply(lambda s: s.shift(1).expanding().mean())
    running_std = grouped.apply(lambda s: s.shift(1).expanding().std())

    running_mean = running_mean.reset_index(level=0, drop=True)
    running_std = running_std.reset_index(level=0, drop=True)

    df = df.reset_index(drop=True)
    running_mean = running_mean.reset_index(drop=True)
    running_std = running_std.reset_index(drop=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (df[AMOUNT_COL] - running_mean) / running_std
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["amt_zscore_vs_card_history"] = z
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper applying every feature function above, in order."""
    df = add_distance_feature(df)
    df = add_datetime_features(df)
    df = add_cyclical_time_features(df)
    df = add_customer_age(df)
    df = add_log_population(df)
    df = add_velocity_features(df)
    df = add_spending_deviation(df)
    return df
