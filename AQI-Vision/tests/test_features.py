"""Tests for src/features.py — verify no leakage and correct shapes."""

import numpy as np
import pandas as pd

import config
from src import features


def test_lag_feature_matches_shifted_target(sample_hourly):
    """aqi_lag_1 for a city must equal that city's AQI shifted by one hour."""
    out = features.add_lag_features(sample_hourly)
    delhi = out[out[config.CITY_COL] == "Delhi"].reset_index(drop=True)
    # Row i's lag_1 should equal row i-1's actual AQI.
    np.testing.assert_allclose(
        delhi["aqi_lag_1"].iloc[1:10].to_numpy(),
        delhi[config.TARGET_COL].iloc[0:9].to_numpy(),
    )


def test_rolling_mean_excludes_current_row(sample_hourly):
    """Rolling mean must use shift(1) so it never includes the current AQI."""
    out = features.add_lag_features(sample_hourly)
    delhi = out[out[config.CITY_COL] == "Delhi"].reset_index(drop=True)
    # rollmean_3 at row 3 = mean of actual AQI rows 0,1,2 (current excluded).
    expected = delhi[config.TARGET_COL].iloc[0:3].mean()
    assert np.isclose(delhi["aqi_rollmean_3"].iloc[3], expected)


def test_temporal_features_present_and_flagged(sample_hourly):
    """Festival + weekend flags and cyclical encodings must be created."""
    out = features.add_temporal_features(sample_hourly)
    for col in ["hour_sin", "hour_cos", "is_weekend", "is_festival_season"]:
        assert col in out.columns
    assert set(out["is_weekend"].unique()).issubset({0, 1})


def test_build_features_drops_nan_rows(sample_hourly):
    """Final feature frame must contain no NaNs (early lag rows dropped)."""
    feat = features.build_features(sample_hourly)
    assert feat.isna().sum().sum() == 0
    assert len(features.get_feature_columns(feat)) > 10
