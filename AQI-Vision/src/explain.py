"""SHAP-based explainability.

SHAP (SHapley Additive exPlanations) attributes each prediction to its features
using a game-theoretic allocation of "credit". Unlike a tree's built-in
`feature_importances_` (which counts split frequency/gain globally and can be
biased toward high-cardinality features), SHAP gives **consistent, per-row**
contributions that sum to the prediction — so we can explain *one* forecast, not
just the model on average.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap

import config


def compute_shap_values(model, X: pd.DataFrame, model_type: str = "tree"):
    """Compute SHAP values with the explainer that matches the model.

    Args:
        model: fitted estimator.
        X: feature matrix to explain.
        model_type: "tree" (XGB/LGBM/RF) or "linear" (Ridge).

    Returns:
        A shap.Explanation object.

    Raises:
        ValueError: for an unknown model_type.
    """
    if model_type == "tree":
        # TreeExplainer is exact and fast for tree ensembles.
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return explainer(X)


def get_top_features(shap_values, n: int = 10) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value (global importance).

    Args:
        shap_values: a shap.Explanation.
        n: number of top features to return.

    Returns:
        DataFrame [feature, mean_abs_shap] sorted descending.

    Note:
        Mean |SHAP| is preferred over the model's built-in importance because it
        is computed on actual predictions and is comparable across model types.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    return df


def plot_shap_summary(shap_values, max_display: int = 15):
    """Beeswarm summary plot (global view of feature effects).

    Args:
        shap_values: a shap.Explanation.
        max_display: number of features to show.

    Returns:
        The matplotlib figure SHAP draws into.
    """
    import matplotlib.pyplot as plt

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    return plt.gcf()


def plot_shap_waterfall(shap_values, idx: int):
    """Waterfall plot explaining a single prediction.

    Args:
        shap_values: a shap.Explanation.
        idx: row index to explain.

    Returns:
        The matplotlib figure. Shows how each feature pushed this one prediction
        above/below the baseline — the literal answer to "why this AQI?".
    """
    import matplotlib.pyplot as plt

    shap.plots.waterfall(shap_values[idx], show=False)
    return plt.gcf()


def plot_shap_dependence(shap_values, feature: str):
    """Dependence plot: feature value vs its SHAP contribution.

    Args:
        shap_values: a shap.Explanation.
        feature: feature name to plot.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    shap.plots.scatter(shap_values[:, feature], show=False)
    return plt.gcf()


def save_shap_values(shap_values, path=config.SHAP_VALUES_FILE) -> None:
    """Persist computed SHAP values for fast dashboard loading."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(shap_values, path)


def load_shap_values(path=config.SHAP_VALUES_FILE):
    """Load persisted SHAP values."""
    return joblib.load(path)
