"""Synthetic CPCB-style AQI data generator.

Why this exists: the real Kaggle dataset is ~1.4M rows and must be downloaded
manually (see data/README.md). To let the *entire* pipeline run, train, and be
demoed without that download, this script fabricates a dataset with the SAME
schema and realistic statistical structure:

  * city-specific baseline pollution levels (Delhi >> Shillong),
  * a yearly seasonal cycle (winter worse than monsoon in North India),
  * a daily cycle (rush-hour morning/evening peaks),
  * a Diwali (Oct/Nov) spike,
  * weekend dips,
  * autocorrelated noise (today looks like yesterday).

The synthetic data is clearly *not* real and is only for plumbing/demo. Real
metrics must come from the real dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Per-city baseline AQI. Rough, opinionated values reflecting known reality.
CITY_BASELINE = {
    "Delhi": 210, "Gurugram": 190, "Lucknow": 170, "Patna": 165, "Kolkata": 150,
    "Ahmedabad": 130, "Jaipur": 130, "Mumbai": 110, "Hyderabad": 100,
    "Bhopal": 110, "Chandigarh": 105, "Amritsar": 120, "Brajrajnagar": 115,
    "Jorapokhar": 120, "Talcher": 115, "Visakhapatnam": 95, "Chennai": 90,
    "Bengaluru": 85, "Amaravati": 90, "Guwahati": 100, "Coimbatore": 75,
    "Kochi": 70, "Ernakulam": 70, "Thiruvananthapuram": 65, "Aizawl": 45,
    "Shillong": 50,
}

# Seasonal multiplier by month (1=Jan). Winter inversion traps pollutants in the
# north; monsoon rain scavenges particulates. Diwali bump folded into Oct/Nov.
MONTH_MULTIPLIER = {
    1: 1.45, 2: 1.25, 3: 1.05, 4: 0.95, 5: 0.90, 6: 0.70,
    7: 0.60, 8: 0.62, 9: 0.75, 10: 1.30, 11: 1.55, 12: 1.50,
}

# Hour-of-day multiplier: morning (8-10) and evening (19-22) traffic peaks.
HOUR_MULTIPLIER = np.array([
    0.85, 0.80, 0.78, 0.78, 0.82, 0.90, 1.05, 1.20, 1.30, 1.25, 1.10, 1.00,
    0.95, 0.92, 0.90, 0.92, 0.98, 1.10, 1.25, 1.30, 1.25, 1.15, 1.00, 0.90,
])


def _aqi_to_pollutants(aqi: np.ndarray, rng: np.random.Generator) -> dict:
    """Back-out plausible pollutant concentrations from a target AQI series.

    In reality AQI is computed *from* pollutants; here we invert that for
    synthetic data. PM2.5/PM10 track AQI closely; gases are noisier.

    Args:
        aqi: array of AQI values.
        rng: numpy random generator.

    Returns:
        Dict of column name -> concentration array.
    """
    n = len(aqi)
    scale = aqi / 200.0
    return {
        "PM2.5": np.clip(aqi * 0.45 + rng.normal(0, 8, n), 1, None),
        "PM10": np.clip(aqi * 0.80 + rng.normal(0, 15, n), 1, None),
        "NO": np.clip(scale * 25 + rng.normal(0, 5, n), 0, None),
        "NO2": np.clip(scale * 35 + rng.normal(0, 7, n), 0, None),
        "NOx": np.clip(scale * 45 + rng.normal(0, 9, n), 0, None),
        "NH3": np.clip(scale * 20 + rng.normal(0, 6, n), 0, None),
        "CO": np.clip(scale * 1.2 + rng.normal(0, 0.3, n), 0, None),
        "SO2": np.clip(scale * 15 + rng.normal(0, 4, n), 0, None),
        "O3": np.clip(scale * 30 + rng.normal(0, 10, n), 0, None),
        "Benzene": np.clip(scale * 3 + rng.normal(0, 1, n), 0, None),
        "Toluene": np.clip(scale * 8 + rng.normal(0, 2, n), 0, None),
        "Xylene": np.clip(scale * 2 + rng.normal(0, 0.8, n), 0, None),
    }


def generate_city_hour(start: str, end: str, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic hourly multi-city dataset.

    Args:
        start: start date, e.g. "2018-01-01".
        end: end date (exclusive), e.g. "2020-01-01".
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame matching the real city_hour.csv schema.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    frames = []

    for city, baseline in CITY_BASELINE.items():
        n = len(timestamps)
        months = timestamps.month.to_numpy()
        hours = timestamps.hour.to_numpy()
        dow = timestamps.dayofweek.to_numpy()

        seasonal = np.array([MONTH_MULTIPLIER[m] for m in months])
        daily = HOUR_MULTIPLIER[hours]
        weekend = np.where(dow >= 5, 0.92, 1.0)  # slightly cleaner on weekends

        # AR(1) noise so consecutive hours are correlated (realistic for AQI).
        noise = np.zeros(n)
        eps = rng.normal(0, baseline * 0.12, n)
        for i in range(1, n):
            noise[i] = 0.85 * noise[i - 1] + eps[i]

        aqi = baseline * seasonal * daily * weekend + noise
        aqi = np.clip(aqi, 10, 500)

        df = pd.DataFrame({config.DATE_COL: timestamps})
        df[config.CITY_COL] = city
        for col, vals in _aqi_to_pollutants(aqi, rng).items():
            df[col] = np.round(vals, 2)
        df[config.TARGET_COL] = np.round(aqi, 0)

        # Inject realistic missingness (~5% of rows have gaps) so preprocessing
        # has something to do.
        mask = rng.random(n) < 0.05
        df.loc[mask, config.POLLUTANT_COLS] = np.nan
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic AQI data.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2020-01-01")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_city_hour(args.start, args.end, args.seed)
    df.to_csv(config.CITY_HOUR_FILE, index=False)

    # Also derive a daily file by averaging, mirroring the real city_day.csv.
    daily = (
        df.assign(Date=pd.to_datetime(df[config.DATE_COL]).dt.date)
        .groupby([config.CITY_COL, "Date"], as_index=False)[
            config.POLLUTANT_COLS + [config.TARGET_COL]
        ]
        .mean(numeric_only=True)
    )
    daily.to_csv(config.CITY_DAY_FILE, index=False)

    print(f"Wrote {len(df):,} hourly rows -> {config.CITY_HOUR_FILE}")
    print(f"Wrote {len(daily):,} daily rows  -> {config.CITY_DAY_FILE}")
    print("NOTE: synthetic data. Use the real Kaggle dataset for reportable metrics.")


if __name__ == "__main__":
    main()
