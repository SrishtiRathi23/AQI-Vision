"""Data ingestion: load the CPCB CSVs, validate the schema, summarise.

Loading is deliberately separated from cleaning so that a schema problem (wrong
file, renamed column) fails loudly and early, before any modelling logic runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def validate_schema(df: pd.DataFrame, expected_cols: list[str]) -> None:
    """Raise a descriptive error if any expected column is missing.

    Args:
        df: DataFrame to check.
        expected_cols: columns that must be present.

    Raises:
        ValueError: listing exactly which columns are missing.
    """
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def load_city_hour(path: str | Path = config.CITY_HOUR_FILE) -> pd.DataFrame:
    """Load the hourly dataset and parse the timestamp.

    Args:
        path: path to city_hour.csv.

    Returns:
        DataFrame sorted by (City, Datetime) with a parsed Datetime column.
    """
    df = pd.read_csv(path)
    validate_schema(df, [config.CITY_COL, config.DATE_COL, config.TARGET_COL])
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    # ML REASON: time series logic (lags, rolling windows) is only valid on a
    # monotonically increasing timeline within each city.
    df = df.sort_values([config.CITY_COL, config.DATE_COL]).reset_index(drop=True)
    return df


def load_city_day(path: str | Path = config.CITY_DAY_FILE) -> pd.DataFrame:
    """Load the daily dataset and parse the date.

    Args:
        path: path to city_day.csv.

    Returns:
        DataFrame sorted by (City, Date).
    """
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else config.DATE_COL
    validate_schema(df, [config.CITY_COL, date_col, config.TARGET_COL])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([config.CITY_COL, date_col]).reset_index(drop=True)
    return df


def summarize_dataset(df: pd.DataFrame, date_col: str = config.DATE_COL) -> dict:
    """Print and return a quick health summary of the dataset.

    Args:
        df: dataset.
        date_col: name of the timestamp column.

    Returns:
        Dict with shape, date range, city count and null fractions.
    """
    null_frac = (df.isna().mean().sort_values(ascending=False)).round(4)
    summary = {
        "rows": len(df),
        "cols": df.shape[1],
        "n_cities": df[config.CITY_COL].nunique(),
        "date_min": str(df[date_col].min()),
        "date_max": str(df[date_col].max()),
        "top_null_columns": null_frac[null_frac > 0].head(10).to_dict(),
    }
    print("=" * 60)
    print("DATASET SUMMARY")
    print(f"  Rows x Cols : {summary['rows']:,} x {summary['cols']}")
    print(f"  Cities      : {summary['n_cities']}")
    print(f"  Date range  : {summary['date_min']} -> {summary['date_max']}")
    print(f"  Null columns: {summary['top_null_columns']}")
    print("=" * 60)
    return summary


def load_or_generate() -> pd.DataFrame:
    """Load real hourly data if present, else fall back to synthetic.

    Returns:
        Hourly DataFrame ready for preprocessing.
    """
    if config.CITY_HOUR_FILE.exists():
        print(f"Loading real dataset: {config.CITY_HOUR_FILE}")
        return load_city_hour()

    print("Real dataset not found -> generating synthetic data for demo.")
    from data.generate_sample import generate_city_hour  # local import

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_city_hour("2018-01-01", "2020-01-01")
    df.to_csv(config.CITY_HOUR_FILE, index=False)
    return load_city_hour()
