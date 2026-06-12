"""End-to-end training pipeline for AQI-Vision.

Run:  python train.py

Steps: load (real or synthetic) -> clean -> engineer features -> temporal split
-> scale -> train 6 models (MLflow-tracked) -> compare -> save best -> SHAP.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import ingestion, preprocessing, features, models, explain
from src.evaluate import compare_models


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load -------------------------------------------------------------- #
    df = ingestion.load_or_generate()
    ingestion.summarize_dataset(df)

    # 2. Clean ------------------------------------------------------------- #
    df = preprocessing.handle_missing_values(df)
    df = preprocessing.drop_long_gaps(df)
    df = preprocessing.remove_outliers(df, config.POLLUTANT_COLS + [config.TARGET_COL])

    # 3. Feature engineering ---------------------------------------------- #
    print("Engineering features...")
    feat = features.build_features(df)
    feature_cols = features.get_feature_columns(feat)
    print(f"  {len(feature_cols)} feature columns, {len(feat):,} usable rows")
    feat.to_parquet(config.PROCESSED_FILE)

    # 4. Temporal split ---------------------------------------------------- #
    train_df, test_df = preprocessing.train_test_split_temporal(feat)

    # 5. Scale ------------------------------------------------------------- #
    # ML REASON: fit the scaler on TRAIN ONLY, then apply to test. Fitting on the
    # full set would leak test-set statistics into training.
    train_scaled, scaler = preprocessing.normalize_features(
        train_df.copy(), feature_cols, method="standard"
    )
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    X_train, y_train = train_scaled[feature_cols], train_scaled[config.TARGET_COL]
    X_test, y_test = test_scaled[feature_cols], test_scaled[config.TARGET_COL]

    # 6-7. Train + evaluate ------------------------------------------------ #
    # NOTE: X_test is SCALED (for the ML models), but the persistence baseline
    # and SARIMA need the UNSCALED test frame so they compare like-for-like with
    # the unscaled AQI target. Passing test_scaled here would scale aqi_lag_1 to
    # ~0 mean and make the baseline look artificially terrible.
    results, fitted = models.train_all_models(
        X_train, y_train, X_test, y_test, test_df
    )

    # 8. Compare ----------------------------------------------------------- #
    table = compare_models(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by RMSE)")
    print(table.to_string())
    print("=" * 60)

    # 9. Save best (excluding the non-parametric baselines) ---------------- #
    trainable = {k: v for k, v in results.items()
                 if k not in ("Persistence", "SARIMA") and fitted.get(k) is not None}
    best_name = min(trainable, key=lambda k: trainable[k]["RMSE"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (RMSE={results[best_name]['RMSE']:.3f})")

    models.save_model(best_model, config.BEST_MODEL_FILE)
    models.save_model(scaler, config.SCALER_FILE)
    with open(config.FEATURE_LIST_FILE, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 10. SHAP on the best model ------------------------------------------ #
    print("Computing SHAP values for the best model...")
    model_type = "linear" if best_name == "Ridge" else "tree"
    # Sample for speed on large test sets.
    sample = X_test.sample(min(2000, len(X_test)), random_state=config.RANDOM_STATE)
    shap_values = explain.compute_shap_values(best_model, sample, model_type)
    explain.save_shap_values(shap_values)
    top = explain.get_top_features(shap_values, n=3)

    # Final summary -------------------------------------------------------- #
    print("\n" + "#" * 60)
    print("FINAL SUMMARY")
    print(f"  Best model : {best_name}")
    print(f"  Test RMSE  : {results[best_name]['RMSE']:.3f}")
    print(f"  Test MAE   : {results[best_name]['MAE']:.3f}")
    print(f"  Test R2    : {results[best_name]['R2']:.3f}")
    print(f"  Top-3 SHAP : {list(top['feature'])}")
    print("#" * 60)
    print("\nThese are YOUR numbers — use exactly these on the application form.")


if __name__ == "__main__":
    main()
