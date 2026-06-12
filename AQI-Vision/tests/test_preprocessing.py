"""Tests for src/preprocessing.py — exercise actual logic, not just imports."""

import numpy as np
import pandas as pd

import config
from src import preprocessing


def test_temporal_split_is_chronological(sample_hourly):
    """Train must end before test begins (no future leakage)."""
    train, test = preprocessing.train_test_split_temporal(sample_hourly, test_size=0.25)
    assert train[config.DATE_COL].max() <= test[config.DATE_COL].min()
    assert len(train) + len(test) == len(sample_hourly)


def test_outlier_capping_bounds_values(sample_hourly):
    """Capping must shrink the range and never drop rows."""
    df = sample_hourly.copy()
    df.loc[0, "PM10"] = 100000  # extreme spike
    capped = preprocessing.remove_outliers(df, ["PM10"])
    assert len(capped) == len(df)               # rows preserved
    assert capped["PM10"].max() < 100000        # spike capped


def test_cyclical_encoding_is_on_unit_circle(sample_hourly):
    """sin^2 + cos^2 must equal 1 for every encoded value."""
    df = sample_hourly.assign(hour=sample_hourly[config.DATE_COL].dt.hour)
    out = preprocessing.encode_cyclical(df, "hour", 24)
    unit = out["hour_sin"] ** 2 + out["hour_cos"] ** 2
    assert np.allclose(unit, 1.0)


def test_missing_values_filled_within_limit(sample_hourly):
    """Short gaps should be imputed; the target should have no NaNs introduced."""
    filled = preprocessing.handle_missing_values(sample_hourly)
    # The injected PM2.5 NaNs (single isolated hours) must be filled.
    assert filled["PM2.5"].isna().sum() < sample_hourly["PM2.5"].isna().sum()
