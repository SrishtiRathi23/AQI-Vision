# 🌫️ AQI-Vision

Hyperlocal Air Quality Forecasting & Health Risk Intelligence for Indian Cities.
AQI-Vision ingests CPCB pollutant data for 26 cities, engineers temporal / lag /
chemistry / city features, benchmarks six models (persistence → Ridge → Random
Forest → XGBoost → LightGBM → SARIMA) with leakage-safe time-series CV, explains
predictions with SHAP, and serves 24–72h forecasts + CPCB health advisories in a
Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-green)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-success)
![SHAP](https://img.shields.io/badge/SHAP-explainable-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)

## Architecture

```
            ┌──────────────┐
 CPCB CSV → │  ingestion   │  load + schema validation
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ preprocessing│  gap-aware imputation, IQR capping,
            └──────┬───────┘  cyclical encoding, temporal split, scaling
                   ▼
            ┌──────────────┐
            │   features   │  temporal · lag/rolling/EWMA · pollutant · city
            └──────┬───────┘
                   ▼
            ┌──────────────┐     ┌─────────────┐
            │    models    │ ──► │   MLflow    │  6 models, TimeSeriesSplit CV
            └──────┬───────┘     └─────────────┘
                   ▼
        ┌──────────┴──────────┐
        ▼                     ▼
  ┌──────────┐          ┌──────────┐
  │ evaluate │          │ explain  │  SHAP (global + per-prediction)
  └────┬─────┘          └────┬─────┘
       └────────┬────────────┘
                ▼
        ┌──────────────┐   ┌─────────────┐
        │   forecast   │ + │ health_risk │  24/48/72h + CPCB advisories
        └──────┬───────┘   └──────┬──────┘
               └────────┬─────────┘
                        ▼
                 ┌──────────────┐
                 │   app.py     │  Streamlit (5 pages) → Cloud Run
                 └──────────────┘
```

## Dataset

CPCB "Air Quality Data in India (2015–2020)" — see [`data/README.md`](data/README.md).
Place `city_day.csv` and `city_hour.csv` in `data/raw/`, or run the synthetic
generator for a demo.

## Quickstart (real data)

```bash
pip install -r requirements.txt
# put city_hour.csv in data/raw/  (see data/README.md)
python train.py            # trains all models, saves best + SHAP
streamlit run app.py       # opens the dashboard
```

## Quickstart (synthetic demo — no download)

```bash
pip install -r requirements.txt
python data/generate_sample.py   # fabricate schema-identical data
python train.py
streamlit run app.py
```

Run the tests with `pytest -q`. Inspect experiments with `mlflow ui`.

## Model comparison

> ⚠️ **Placeholders** — replace with the numbers `python train.py` prints on the
> **real** dataset before quoting them anywhere.

| Rank | Model        | RMSE | MAE  | R²   |
|------|--------------|------|------|------|
| 1    | XGBoost      | _tbd_| _tbd_| _tbd_|
| 2    | LightGBM     | _tbd_| _tbd_| _tbd_|
| 3    | RandomForest | _tbd_| _tbd_| _tbd_|
| 4    | Ridge        | _tbd_| _tbd_| _tbd_|
| 5    | SARIMA       | _tbd_| _tbd_| _tbd_|
| 6    | Persistence  | _tbd_| _tbd_| _tbd_|

## Key findings (fill from your run)

- Northern cities show a strong winter spike vs monsoon trough (seasonal cycle).
- AQI lag features (esp. `aqi_lag_24`, `aqi_rollmean_24`) dominate SHAP importance.
- `is_festival_season` lifts predicted AQI around Diwali.

## Tech stack

Python · pandas/NumPy · scikit-learn · XGBoost · LightGBM · statsmodels · SHAP ·
Plotly/Matplotlib/Seaborn · MLflow · Streamlit · Docker · Google Cloud Run.

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md). Live demo: _tbd_.
