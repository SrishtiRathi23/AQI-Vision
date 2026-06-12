"""Forecasting: build future feature vectors and project AQI ahead.

The model is trained on engineered features (including AQI lags). To forecast h
hours ahead we roll forward iteratively: predict the next hour, append it to the
history, recompute lag/rolling features, and repeat. Uncertainty bands come from
bootstrap resampling of the model's recent residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features import build_features, get_feature_columns


def _recompute_features_for_city(history: pd.DataFrame) -> pd.DataFrame:
    """Rebuild engineered features for a single city's history.

    Args:
        history: rows for one city, time-sorted, with raw pollutant columns.

    Returns:
        Feature frame for that city.
    """
    return build_features(history)


def forecast_aqi(
    model, scaler, feature_cols: list[str], history: pd.DataFrame,
    city: str, hours_ahead: int = 72,
) -> pd.DataFrame:
    """Iteratively forecast AQI for one city.

    Args:
        model: fitted estimator (must implement predict).
        scaler: fitted scaler for the feature columns (or None).
        feature_cols: ordered model input columns.
        history: recent hourly history for `city` (raw schema).
        city: city to forecast.
        hours_ahead: horizon length.

    Returns:
        DataFrame [Datetime, City, forecast] of length hours_ahead.
    """
    # Strip to RAW columns only. `history` may already carry engineered columns
    # (it often comes from the processed feature table); rebuilding features on
    # top of those would duplicate one-hot columns like season_Summer.
    raw_cols = [config.DATE_COL, config.CITY_COL] + config.POLLUTANT_COLS + [config.TARGET_COL]
    hist = history[history[config.CITY_COL] == city].copy()
    hist = hist[[c for c in raw_cols if c in hist.columns]]
    hist = hist.sort_values(config.DATE_COL).reset_index(drop=True)
    last_time = hist[config.DATE_COL].max()

    preds = []
    for step in range(1, hours_ahead + 1):
        feats = _recompute_features_for_city(hist)
        if feats.empty:
            break
        # Reindex to the exact training feature set. A short forecast window may
        # not contain every season, so some one-hot columns (e.g. season_Winter)
        # can be missing; fill those with 0 — they legitimately mean "not that
        # season" — and keep column order identical to training.
        row = feats.iloc[[-1]].reindex(columns=feature_cols, fill_value=0)
        X = scaler.transform(row) if scaler is not None else row.to_numpy()
        yhat = float(model.predict(X)[0])
        yhat = float(np.clip(yhat, 0, 500))

        next_time = last_time + pd.Timedelta(hours=step)
        preds.append({config.DATE_COL: next_time, config.CITY_COL: city,
                      "forecast": yhat})

        # Append the prediction as new "observed" AQI so the next iteration's
        # lag features see it. Pollutants are carried forward (persistence) —
        # ML REASON: we don't have future pollutant readings, so the AQI lags do
        # the heavy lifting for short horizons.
        # Use a DataFrame slice (not Series.to_frame().T) so column dtypes are
        # preserved — otherwise Datetime would be coerced to object and the next
        # iteration's .dt accessor would fail.
        new_row = hist.iloc[[-1]].copy()
        new_row[config.DATE_COL] = next_time
        new_row[config.TARGET_COL] = yhat
        hist = pd.concat([hist, new_row], ignore_index=True)

    return pd.DataFrame(preds)


def bootstrap_intervals(
    residuals: np.ndarray, point_forecast: np.ndarray,
    n_samples: int = config.BOOTSTRAP_SAMPLES, alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build prediction-interval bands by resampling residuals.

    Args:
        residuals: in-sample residuals (y_true - y_pred) of the model.
        point_forecast: the point forecast to wrap with a band.
        n_samples: number of bootstrap draws.
        alpha: tail mass (0.10 -> 90% interval).

    Returns:
        (lower, upper) arrays the same length as point_forecast.

    Note:
        Resampling residuals avoids assuming Gaussian errors; the band reflects
        the model's actual error distribution.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    draws = np.empty((n_samples, len(point_forecast)))
    for i in range(n_samples):
        sampled = rng.choice(residuals, size=len(point_forecast), replace=True)
        draws[i] = point_forecast + sampled
    lower = np.percentile(draws, 100 * alpha / 2, axis=0)
    upper = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    return np.clip(lower, 0, 500), np.clip(upper, 0, 500)


def plot_forecast(forecast_df: pd.DataFrame, lower=None, upper=None, city: str = ""):
    """Plot the forecast line with an optional uncertainty band.

    Args:
        forecast_df: DataFrame with Datetime and forecast columns.
        lower: optional lower band array.
        upper: optional upper band array.
        city: city label for the title.

    Returns:
        Plotly figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    t = forecast_df[config.DATE_COL]
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(x=list(t) + list(t[::-1]),
                                 y=list(upper) + list(lower[::-1]),
                                 fill="toself", fillcolor="rgba(0,120,255,0.15)",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 name="90% interval"))
    fig.add_trace(go.Scatter(x=t, y=forecast_df["forecast"], mode="lines+markers",
                             name="Forecast", line=dict(color="#0078ff")))
    fig.update_layout(title=f"AQI forecast — {city}", xaxis_title="Time",
                      yaxis_title="AQI")
    return fig
