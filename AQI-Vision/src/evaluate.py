"""Evaluation metrics and comparison/diagnostic plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the standard regression metric set.

    Args:
        y_true: observed AQI.
        y_pred: predicted AQI.

    Returns:
        Dict with RMSE, MAE, MAPE (%) and R².
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # Guard against division by zero in MAPE.
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a ranked comparison table from per-model metric dicts.

    Args:
        results: {model_name: metric_dict}.

    Returns:
        DataFrame sorted by RMSE ascending (best first).
    """
    table = pd.DataFrame(results).T
    table = table.sort_values("RMSE")
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.round(4)


def plot_predictions(y_true, y_pred, model_name: str, n: int = 300) -> go.Figure:
    """Overlay observed vs predicted for the first n test points.

    Args:
        y_true: observed values.
        y_pred: predicted values.
        model_name: label for the title.
        n: number of points to draw.

    Returns:
        Plotly figure.
    """
    y_true = np.asarray(y_true)[:n]
    y_pred = np.asarray(y_pred)[:n]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(title=f"{model_name}: Actual vs Predicted (first {n})",
                      xaxis_title="Test step", yaxis_title="AQI")
    return fig


def plot_residuals(y_true, y_pred, model_name: str) -> go.Figure:
    """Residual histogram + Normal Q-Q plot.

    A well-behaved model has roughly symmetric, mean-zero residuals that fall on
    the Q-Q line — a quick check for bias and heavy tails.

    Args:
        y_true: observed values.
        y_pred: predicted values.
        model_name: label for the title.

    Returns:
        Plotly figure with two panels.
    """
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residual distribution", "Normal Q-Q"))
    fig.add_trace(go.Histogram(x=resid, nbinsx=50, name="Residuals"), row=1, col=1)

    (osm, osr), _ = stats.probplot(resid, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residuals"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=osm, mode="lines", name="Ideal"), row=1, col=2)
    fig.update_layout(title=f"{model_name}: Residual diagnostics", showlegend=False)
    return fig


def plot_model_comparison_bar(results: dict[str, dict]) -> go.Figure:
    """Grouped bar chart of RMSE/MAE/MAPE across models.

    Args:
        results: {model_name: metric_dict}.

    Returns:
        Plotly figure (R² excluded here since it is on a different scale).
    """
    models = list(results.keys())
    fig = go.Figure()
    for metric in ["RMSE", "MAE", "MAPE"]:
        fig.add_trace(go.Bar(name=metric, x=models,
                             y=[results[m][metric] for m in models]))
    fig.update_layout(barmode="group", title="Model comparison (lower is better)",
                      yaxis_title="Error")
    return fig
