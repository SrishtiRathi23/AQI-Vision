"""AQI-Vision Streamlit dashboard.

Five pages: City Overview, Forecast, Model Performance, Explainability (SHAP),
and EDA. The app loads the processed feature table and the trained best model;
if either is missing it tells the user to run `python train.py` first.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import models, explain, forecast
from src.evaluate import compute_metrics
from src.health_risk import (classify_health_risk, generate_health_advisory,
                             compute_risk_trend, VALID_SEGMENTS)

st.set_page_config(page_title="AQI-Vision", page_icon="🌫️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame | None:
    if config.PROCESSED_FILE.exists():
        return pd.read_parquet(config.PROCESSED_FILE)
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts = {}
    if config.BEST_MODEL_FILE.exists():
        artifacts["model"] = models.load_model(config.BEST_MODEL_FILE)
    if config.SCALER_FILE.exists():
        artifacts["scaler"] = models.load_model(config.SCALER_FILE)
    if config.FEATURE_LIST_FILE.exists():
        artifacts["features"] = json.loads(config.FEATURE_LIST_FILE.read_text())
    if config.SHAP_VALUES_FILE.exists():
        artifacts["shap"] = explain.load_shap_values()
    return artifacts


def aqi_color(value: float) -> str:
    return classify_health_risk(value)["color"]


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🌫️ AQI-Vision")
    st.caption("Hyperlocal Air Quality Forecasting & Health Risk Intelligence for Indian Cities")

    feat = load_features()
    if feat is None:
        st.error("No processed data found. Run `python train.py` first.")
        st.stop()

    artifacts = load_artifacts()
    cities = sorted(feat[config.CITY_COL].unique())

    page = st.sidebar.radio(
        "Page",
        ["City Overview", "Forecast", "Model Performance", "Explainability (SHAP)", "EDA"],
    )

    if page == "City Overview":
        _page_overview(feat, cities)
    elif page == "Forecast":
        _page_forecast(feat, cities, artifacts)
    elif page == "Model Performance":
        _page_performance(feat, artifacts)
    elif page == "Explainability (SHAP)":
        _page_explain(artifacts)
    else:
        _page_eda(feat, cities)


# --------------------------------------------------------------------------- #
# Page 1 — City Overview
# --------------------------------------------------------------------------- #
def _page_overview(feat, cities):
    st.header("City Overview")
    city = st.selectbox("City", cities)
    cdf = feat[feat[config.CITY_COL] == city].sort_values(config.DATE_COL)

    dmin, dmax = cdf[config.DATE_COL].min(), cdf[config.DATE_COL].max()
    rng = st.date_input("Date range", value=(dmin.date(), dmax.date()),
                        min_value=dmin.date(), max_value=dmax.date())
    if isinstance(rng, tuple) and len(rng) == 2:
        mask = (cdf[config.DATE_COL].dt.date >= rng[0]) & (cdf[config.DATE_COL].dt.date <= rng[1])
        cdf = cdf[mask]

    current = cdf[config.TARGET_COL].iloc[-1] if len(cdf) else np.nan
    week = cdf[config.TARGET_COL].tail(7 * 24).mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current AQI", f"{current:.0f}")
    c2.metric("7-day avg", f"{week:.0f}")
    c3.metric("Worst", f"{cdf[config.TARGET_COL].max():.0f}")
    c4.metric("Best", f"{cdf[config.TARGET_COL].min():.0f}")

    st.plotly_chart(
        px.line(cdf, x=config.DATE_COL, y=config.TARGET_COL,
                title=f"AQI trend — {city}"),
        use_container_width=True,
    )

    # Month x hour heatmap of average AQI.
    cdf = cdf.assign(hour=cdf[config.DATE_COL].dt.hour, month=cdf[config.DATE_COL].dt.month)
    pivot = cdf.pivot_table(index="month", columns="hour",
                            values=config.TARGET_COL, aggfunc="mean")
    st.plotly_chart(
        px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrRd",
                  title="Average AQI by month × hour", labels=dict(color="AQI")),
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Page 2 — Forecast
# --------------------------------------------------------------------------- #
def _page_forecast(feat, cities, artifacts):
    st.header("Forecast")
    if "model" not in artifacts:
        st.warning("No trained model found. Run `python train.py`.")
        return

    city = st.selectbox("City", cities)
    horizon = st.radio("Horizon", config.FORECAST_HORIZONS, horizontal=True,
                       format_func=lambda h: f"{h}h")
    segment = st.selectbox("Population segment", VALID_SEGMENTS)

    history = feat[feat[config.CITY_COL] == city].tail(24 * 30).copy()
    with st.spinner("Forecasting..."):
        fc = forecast.forecast_aqi(artifacts["model"], artifacts.get("scaler"),
                                   artifacts["features"], history, city, horizon)
    if fc.empty:
        st.error("Not enough history to forecast this city.")
        return

    # Bootstrap band from recent residuals (approx: forecast vs lag-1).
    resid = (history[config.TARGET_COL] - history["aqi_lag_1"]).dropna().to_numpy()
    lower, upper = forecast.bootstrap_intervals(resid, fc["forecast"].to_numpy())
    st.plotly_chart(forecast.plot_forecast(fc, lower, upper, city),
                    use_container_width=True)

    peak = fc["forecast"].max()
    risk = classify_health_risk(peak)
    trend = compute_risk_trend(fc)
    st.markdown(
        f"**Peak forecast AQI:** {peak:.0f} — "
        f"<span style='color:{risk['color']};font-weight:bold'>{risk['category']}</span> "
        f"· trend: **{trend}**", unsafe_allow_html=True,
    )
    st.info(f"Advisory ({segment}): {generate_health_advisory(peak, segment)}")


# --------------------------------------------------------------------------- #
# Page 3 — Model Performance
# --------------------------------------------------------------------------- #
def _page_performance(feat, artifacts):
    st.header("Model Performance")
    st.markdown(
        "Run `python train.py` to populate MLflow with all six models. "
        "Open the MLflow UI with `mlflow ui` to compare runs interactively."
    )
    if "model" not in artifacts:
        st.warning("No trained model found.")
        return
    st.success(f"Best model in use: **{type(artifacts['model']).__name__}**")
    st.write("Feature count:", len(artifacts.get("features", [])))


# --------------------------------------------------------------------------- #
# Page 4 — Explainability
# --------------------------------------------------------------------------- #
def _page_explain(artifacts):
    st.header("Explainability (SHAP)")
    if "shap" not in artifacts:
        st.warning("No SHAP values found. Run `python train.py`.")
        return
    sv = artifacts["shap"]

    st.subheader("Global feature importance (mean |SHAP|)")
    top = explain.get_top_features(sv, n=15)
    st.plotly_chart(
        px.bar(top.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
               orientation="h", title="Top features driving AQI"),
        use_container_width=True,
    )

    st.subheader("Explain a single prediction")
    idx = st.slider("Row index", 0, len(sv) - 1, 0)
    contrib = pd.DataFrame({
        "feature": sv.feature_names,
        "shap_value": sv.values[idx],
    }).reindex(np.argsort(np.abs(sv.values[idx]))[::-1]).head(10)
    st.plotly_chart(
        px.bar(contrib, x="shap_value", y="feature", orientation="h",
               color="shap_value", color_continuous_scale="RdBu",
               title="Why this prediction? (top-10 contributions)"),
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Page 5 — EDA
# --------------------------------------------------------------------------- #
def _page_eda(feat, cities):
    st.header("Exploratory Data Analysis")

    st.subheader("AQI distribution by city")
    sample = feat[feat[config.CITY_COL].isin(cities)]
    st.plotly_chart(
        px.violin(sample, x=config.CITY_COL, y=config.TARGET_COL, box=True,
                  title="AQI distribution per city").update_xaxes(tickangle=45),
        use_container_width=True,
    )

    st.subheader("Pollutant correlation heatmap")
    present = [c for c in config.POLLUTANT_COLS + [config.TARGET_COL] if c in feat.columns]
    corr = feat[present].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdBu_r"),
                    use_container_width=True)

    st.subheader("Top 10 most polluted cities (mean AQI)")
    ranking = (feat.groupby(config.CITY_COL)[config.TARGET_COL].mean()
               .sort_values(ascending=False).head(10).reset_index())
    st.plotly_chart(
        px.bar(ranking, x=config.TARGET_COL, y=config.CITY_COL, orientation="h",
               color=config.TARGET_COL, color_continuous_scale="YlOrRd"),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
