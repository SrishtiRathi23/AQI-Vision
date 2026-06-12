"""Preprocessing: missing-value handling, outlier capping, scaling, splitting.

Each function is written for *time series* data, where the order of rows carries
information. That constraint drives several non-obvious choices flagged below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config


def handle_missing_values(df: pd.DataFrame, group_col: str = config.CITY_COL) -> pd.DataFrame:
    """Impute missing pollutant values with a gap-length-aware strategy.

    Strategy, applied per city so one city's data never leaks into another's:
      * short gaps (<= SHORT_GAP_HOURS): forward fill — the last reading is the
        best estimate for a brief sensor dropout.
      * medium gaps (<= MEDIUM_GAP_HOURS): linear interpolation — smooth drift
        is more plausible than a flat hold over several hours.
      * long gaps (> MEDIUM_GAP_HOURS): left as NaN and dropped later — we will
        not invent many hours of data we have no basis for.

    Args:
        df: hourly dataset.
        group_col: column to group by (city).

    Returns:
        DataFrame with short/medium gaps filled; long gaps still NaN.
    """
    cols = [c for c in config.POLLUTANT_COLS + [config.TARGET_COL] if c in df.columns]
    out = df.copy()

    def _fill(group: pd.DataFrame) -> pd.DataFrame:
        # ML REASON: limit= caps how far a fill propagates so a long outage is
        # NOT silently filled with stale/interpolated values.
        ff = group[cols].ffill(limit=config.SHORT_GAP_HOURS)
        interp = ff.interpolate(method="linear", limit=config.MEDIUM_GAP_HOURS)
        group[cols] = interp
        return group

    out = out.groupby(group_col, group_keys=False)[out.columns].apply(_fill)
    return out


def drop_long_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the target is still missing after imputation.

    Args:
        df: dataset post-imputation.

    Returns:
        DataFrame with no missing target.
    """
    return df.dropna(subset=[config.TARGET_COL]).reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame, cols: list[str], method: str = "iqr"
) -> pd.DataFrame:
    """Cap (winsorize) outliers rather than delete the rows.

    Args:
        df: dataset.
        cols: numeric columns to treat.
        method: only "iqr" is implemented (Tukey fences).

    Returns:
        DataFrame with values clipped to [Q1 - k*IQR, Q3 + k*IQR].

    Raises:
        ValueError: if an unknown method is requested.
    """
    if method != "iqr":
        raise ValueError(f"Unsupported outlier method: {method}")

    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - config.IQR_MULTIPLIER * iqr
        upper = q3 + config.IQR_MULTIPLIER * iqr
        # ML REASON: we CAP instead of DROP. A genuine pollution spike (Diwali)
        # is signal, not error; deleting it would also break the hourly
        # continuity that lag/rolling features depend on.
        out[col] = out[col].clip(lower, upper)
    return out


def encode_cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Encode a cyclical integer feature as a (sin, cos) pair.

    Args:
        df: dataset.
        col: source column (e.g. hour 0..23).
        period: the cycle length (24 for hour, 12 for month, 7 for weekday).

    Returns:
        DataFrame with `{col}_sin` and `{col}_cos` columns added.
    """
    # ML REASON: one-hot would treat hour 23 and hour 0 as maximally different.
    # sin/cos places them adjacent on a circle, so the model learns that 23:00
    # and 00:00 are neighbours.
    out = df.copy()
    radians = 2.0 * np.pi * out[col] / period
    out[f"{col}_sin"] = np.sin(radians)
    out[f"{col}_cos"] = np.cos(radians)
    return out


def normalize_features(
    df: pd.DataFrame, cols: list[str], method: str = "standard"
) -> tuple[pd.DataFrame, object]:
    """Scale features and return both the transformed frame and the scaler.

    Args:
        df: dataset.
        cols: columns to scale.
        method: "standard" (zero mean/unit var) or "minmax" (to [0,1]).

    Returns:
        (scaled DataFrame, fitted scaler). The scaler MUST be persisted so the
        exact same transform can be applied at inference time.

    Raises:
        ValueError: for an unknown method.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    out = df.copy()
    out[cols] = scaler.fit_transform(out[cols])
    return out, scaler


def train_test_split_temporal(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, date_col: str = config.DATE_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: oldest rows train, newest rows test.

    Args:
        df: dataset.
        test_size: fraction held out as the most-recent tail.
        date_col: timestamp column to sort on.

    Returns:
        (train_df, test_df).
    """
    # ML REASON: a random split would let the model peek at the future to predict
    # the past — leakage that inflates metrics. Forecasting must be evaluated on
    # data that comes strictly *after* the training window.
    ordered = df.sort_values(date_col)
    cutoff = int(len(ordered) * (1.0 - test_size))
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()
