"""Feature engineering.

Grouped into four families, each a separate function so it can be tested and
explained independently:
  1. temporal   — when is it? (hour/day/month cycles, season, festivals)
  2. lag        — what was AQI recently? (lags, rolling stats, EWMA)
  3. pollutant  — chemistry-driven interactions between pollutants
  4. city       — per-city baselines and tier

All lag/rolling features are computed *within each city* and use only past
values, so no future information leaks into a row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.preprocessing import encode_cyclical


# --------------------------------------------------------------------------- #
# 1. Temporal features
# --------------------------------------------------------------------------- #
def add_temporal_features(df: pd.DataFrame, date_col: str = config.DATE_COL) -> pd.DataFrame:
    """Add calendar-derived features.

    Args:
        df: dataset with a datetime column.
        date_col: name of that column.

    Returns:
        DataFrame with hour/day/month cyclical encodings, weekend flag,
        India-specific season and a festival-season flag.
    """
    out = df.copy()
    dt = out[date_col].dt

    out["hour"] = dt.hour
    out["dayofweek"] = dt.dayofweek
    out["month"] = dt.month

    out = encode_cyclical(out, "hour", 24)
    out = encode_cyclical(out, "dayofweek", 7)
    out = encode_cyclical(out, "month", 12)

    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["season"] = out["month"].map(config.SEASON_MAP)
    # ML REASON: Diwali (Oct/Nov) firecrackers cause sharp PM spikes the calendar
    # cycles alone can't express — domain knowledge encoded as an explicit flag.
    out["is_festival_season"] = out["month"].isin(config.FESTIVAL_MONTHS).astype(int)

    # One-hot the 4 seasons (low cardinality, non-cyclical at this granularity).
    out = pd.get_dummies(out, columns=["season"], prefix="season", dtype=int)
    return out


# --------------------------------------------------------------------------- #
# 2. Lag / rolling features
# --------------------------------------------------------------------------- #
def add_lag_features(
    df: pd.DataFrame, target: str = config.TARGET_COL, group_col: str = config.CITY_COL
) -> pd.DataFrame:
    """Add lagged AQI, rolling mean/std and EWMA features.

    Args:
        df: dataset (must be time-sorted within each city).
        target: column to lag (AQI).
        group_col: city column to group by.

    Returns:
        DataFrame with lag/rolling/EWMA columns added.
    """
    out = df.copy()
    grp = out.groupby(group_col)[target]

    # Plain lags: AQI N hours ago. Lag-24 captures "same hour yesterday".
    for lag in config.LAG_HOURS:
        out[f"aqi_lag_{lag}"] = grp.shift(lag)

    # Rolling means/std use .shift(1) first so the current row is excluded —
    # ML REASON: without the shift the window would include the value we are
    # trying to predict, which is target leakage.
    shifted = out.groupby(group_col)[target].shift(1)
    for w in config.ROLLING_MEAN_WINDOWS:
        out[f"aqi_rollmean_{w}"] = shifted.groupby(out[group_col]).transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
    for w in config.ROLLING_STD_WINDOWS:
        # Rolling std captures volatility; high variance precedes unstable AQI.
        out[f"aqi_rollstd_{w}"] = shifted.groupby(out[group_col]).transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()
        )
    for span in config.EWMA_SPANS:
        # EWMA weights recent hours more — reacts faster than a flat mean.
        out[f"aqi_ewma_{span}"] = shifted.groupby(out[group_col]).transform(
            lambda s, span=span: s.ewm(span=span, adjust=False).mean()
        )
    return out


# --------------------------------------------------------------------------- #
# 3. Pollutant interaction features
# --------------------------------------------------------------------------- #
def add_pollutant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add chemistry-motivated pollutant interactions.

    Args:
        df: dataset with raw pollutant columns.

    Returns:
        DataFrame with ratio/sum interaction features (only those whose source
        columns are present).
    """
    out = df.copy()

    if {"PM2.5", "PM10"}.issubset(out.columns):
        # Fine-particle fraction. A high ratio implies combustion sources
        # (vehicles, crop burning) rather than dust.
        out["pm_ratio"] = out["PM2.5"] / (out["PM10"] + 1e-6)
    if {"NO", "NO2"}.issubset(out.columns):
        # Total nitrogen oxides — a direct traffic-emission proxy.
        out["nox_total"] = out["NO"] + out["NO2"]
    if {"O3", "NO2"}.issubset(out.columns):
        # Oxidant index Ox = O3 + NO2, a more stable photochemical indicator
        # than O3 alone (O3 and NO2 trade off through the day).
        out["oxidant_index"] = out["O3"] + out["NO2"]
    return out


# --------------------------------------------------------------------------- #
# 4. City-level features
# --------------------------------------------------------------------------- #
def add_city_features(
    df: pd.DataFrame, target: str = config.TARGET_COL, group_col: str = config.CITY_COL
) -> pd.DataFrame:
    """Add per-city baseline, daily cross-city rank and tier.

    Args:
        df: dataset.
        target: AQI column.
        group_col: city column.

    Returns:
        DataFrame with city_baseline_aqi, city_aqi_rank and city_tier.
    """
    out = df.copy()

    # ML REASON: expanding() mean uses only past rows, so the "historical
    # baseline" for a row never includes that row or any future row (no leakage).
    out["city_baseline_aqi"] = (
        out.groupby(group_col)[target]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # Cross-sectional rank of cities on each timestamp (who is dirtiest now).
    if config.DATE_COL in out.columns:
        out["city_aqi_rank"] = out.groupby(config.DATE_COL)[target].rank(method="dense")

    out["city_tier"] = out[group_col].apply(
        lambda c: 1 if c in config.TIER_1_CITIES else 2
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline in dependency order.

    Args:
        df: cleaned hourly dataset.

    Returns:
        Feature-rich DataFrame. Early rows with NaN lags are dropped.
    """
    out = add_temporal_features(df)
    out = add_pollutant_features(out)
    out = add_city_features(out)
    out = add_lag_features(out)
    # The longest lag/window leaves NaNs at the start of each city series.
    out = out.dropna().reset_index(drop=True)
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (everything numeric except identifiers).

    Args:
        df: feature DataFrame.

    Returns:
        Ordered list of feature column names.
    """
    exclude = {config.TARGET_COL, config.CITY_COL, config.DATE_COL,
               "hour", "dayofweek", "month"}
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in exclude]
