"""Model definitions and training with MLflow experiment tracking.

Six models, in increasing sophistication, so we can show *why* a complex model
is justified rather than assuming it:
  1. Persistence  — naive baseline (predict next = last). Everything must beat it.
  2. Ridge        — linear baseline with L2 regularisation.
  3. RandomForest — bagged trees, captures non-linearity.
  4. XGBoost      — gradient boosting, usually the strongest.
  5. LightGBM     — leaf-wise boosting, faster; direct rival to XGBoost.
  6. SARIMA       — classical statistical time series model.
"""

from __future__ import annotations

import time
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

import config
from src.evaluate import compute_metrics

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def persistence_predict(test_df: pd.DataFrame) -> np.ndarray:
    """Naive forecast: predict each AQI as the previous hour's AQI.

    Args:
        test_df: test slice containing the `aqi_lag_1` feature.

    Returns:
        Array of predictions (= the lag-1 column).
    """
    # ML REASON: a persistence model is the honest bar. If a fancy model can't
    # beat "tomorrow == today", it isn't learning anything useful.
    return test_df["aqi_lag_1"].to_numpy()


# --------------------------------------------------------------------------- #
# Cross-validation helper
# --------------------------------------------------------------------------- #
def _time_series_search(estimator, search_space, X, y):
    """Tune hyperparameters with RandomizedSearchCV over TimeSeriesSplit.

    Args:
        estimator: the model to tune.
        search_space: dict of param distributions.
        X: training features.
        y: training target.

    Returns:
        The best fitted estimator.
    """
    # ML REASON: TimeSeriesSplit trains on past folds and validates on the next
    # contiguous block. Plain KFold would shuffle future rows into training and
    # leak information backwards in time, giving optimistic, invalid scores.
    cv = TimeSeriesSplit(n_splits=config.N_CV_SPLITS)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=config.N_RANDOM_SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    test_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Train, log and evaluate all six models.

    Args:
        X_train, y_train: training features/target.
        X_test, y_test: test features/target.
        test_df: full test frame (needed for the persistence baseline).

    Returns:
        (results, fitted) where results maps model name -> metric dict and
        fitted maps model name -> trained estimator (None for persistence).
    """
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    def _log(name, model, params, train_time):
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics({**metrics, "train_time_s": train_time})
        results[name] = metrics
        fitted[name] = model
        print(f"  {name:14s} RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}"
              f"  ({train_time:.1f}s)")

    print("Training models...")

    # 1. Persistence baseline
    t0 = time.time()
    preds = persistence_predict(test_df)
    metrics = compute_metrics(y_test, preds)
    with mlflow.start_run(run_name="Persistence"):
        mlflow.log_metrics({**metrics, "train_time_s": time.time() - t0})
    results["Persistence"] = metrics
    fitted["Persistence"] = None
    print(f"  {'Persistence':14s} RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}")

    # 2. Ridge
    t0 = time.time()
    ridge = Ridge(**config.RIDGE_PARAMS).fit(X_train, y_train)
    _log("Ridge", ridge, config.RIDGE_PARAMS, time.time() - t0)

    # 3. Random Forest
    t0 = time.time()
    rf = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS).fit(X_train, y_train)
    _log("RandomForest", rf, config.RANDOM_FOREST_PARAMS, time.time() - t0)

    # 4. XGBoost (tuned)
    t0 = time.time()
    xgb_base = XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1,
                            tree_method="hist")
    xgb, xgb_params = _time_series_search(xgb_base, config.XGB_SEARCH_SPACE,
                                          X_train, y_train)
    _log("XGBoost", xgb, xgb_params, time.time() - t0)

    # 5. LightGBM (tuned)
    t0 = time.time()
    lgbm_base = LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgbm, lgbm_params = _time_series_search(lgbm_base, config.LGBM_SEARCH_SPACE,
                                            X_train, y_train)
    _log("LightGBM", lgbm, lgbm_params, time.time() - t0)

    # 6. SARIMA (single representative city to keep runtime sane)
    try:
        sarima_metrics = _train_sarima(test_df, y_test)
        results["SARIMA"] = sarima_metrics
        fitted["SARIMA"] = None  # SARIMA is refit per-series at forecast time
    except Exception as exc:  # statsmodels can be brittle on odd series
        print(f"  SARIMA skipped ({exc})")

    return results, fitted


def _train_sarima(test_df: pd.DataFrame, y_test: pd.Series) -> dict:
    """Fit a SARIMA on one busy city and score it on the test tail.

    Args:
        test_df: test frame (used to pick a city and align the target).
        y_test: aligned test target.

    Returns:
        Metric dict for SARIMA.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # ML REASON: SARIMA is univariate and per-series. Fitting it on every city is
    # slow, so we benchmark it on the highest-volume city as a representative
    # classical baseline against the multivariate ML models.
    city = test_df[config.CITY_COL].value_counts().idxmax()
    series = test_df[test_df[config.CITY_COL] == city][config.TARGET_COL].reset_index(drop=True)
    split = int(len(series) * 0.7)
    train, test = series[:split], series[split:]

    model = SARIMAX(train, order=config.SARIMA_ORDER,
                    seasonal_order=config.SARIMA_SEASONAL_ORDER,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    preds = fit.forecast(steps=len(test))
    metrics = compute_metrics(test.to_numpy(), np.asarray(preds))

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="SARIMA"):
        mlflow.log_params({"order": config.SARIMA_ORDER,
                           "seasonal_order": config.SARIMA_SEASONAL_ORDER,
                           "city": city})
        mlflow.log_metrics(metrics)
    print(f"  {'SARIMA':14s} RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}  ({city})")
    return metrics


def save_model(model, path=config.BEST_MODEL_FILE) -> None:
    """Persist a fitted model to disk.

    Args:
        model: fitted estimator.
        path: output .pkl path.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=config.BEST_MODEL_FILE):
    """Load a persisted model.

    Args:
        path: .pkl path.

    Returns:
        The deserialised estimator.
    """
    return joblib.load(path)
