"""Shared pytest fixtures: a small synthetic hourly frame for fast tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


@pytest.fixture
def sample_hourly():
    """Two cities, 200 hours each, with a few injected NaNs."""
    rng = np.random.default_rng(0)
    frames = []
    times = pd.date_range("2019-01-01", periods=200, freq="h")
    for city, base in [("Delhi", 200), ("Chennai", 90)]:
        df = pd.DataFrame({config.DATE_COL: times})
        df[config.CITY_COL] = city
        aqi = base + rng.normal(0, 15, len(times)).cumsum() * 0.05 + 10
        for col in config.POLLUTANT_COLS:
            df[col] = np.clip(aqi * rng.uniform(0.3, 0.8) + rng.normal(0, 5, len(times)), 1, None)
        df[config.TARGET_COL] = np.clip(aqi, 10, 500)
        df.loc[df.sample(5, random_state=1).index, "PM2.5"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
