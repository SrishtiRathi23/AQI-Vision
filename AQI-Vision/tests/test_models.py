"""Tests for metrics, health-risk logic and the persistence baseline."""

import numpy as np
import pandas as pd
import pytest

import config
from src.evaluate import compute_metrics, compare_models
from src.health_risk import (classify_health_risk, generate_health_advisory,
                             compute_risk_trend)


def test_metrics_perfect_prediction():
    """Perfect predictions -> zero error, R2 == 1."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = compute_metrics(y, y)
    assert m["RMSE"] == 0
    assert m["MAE"] == 0
    assert np.isclose(m["R2"], 1.0)


def test_compare_models_ranks_by_rmse():
    """The lowest-RMSE model must be ranked first."""
    results = {
        "A": {"RMSE": 30, "MAE": 20, "MAPE": 10, "R2": 0.5},
        "B": {"RMSE": 10, "MAE": 5, "MAPE": 4, "R2": 0.9},
    }
    table = compare_models(results)
    assert table.index[0] == "B"
    assert table.iloc[0]["rank"] == 1


def test_health_classification_bands():
    """AQI values must map to the correct CPCB category."""
    assert classify_health_risk(25)["category"] == "Good"
    assert classify_health_risk(150)["category"] == "Moderate"
    assert classify_health_risk(450)["category"] == "Severe"


def test_health_advisory_segment_validation():
    """Unknown segment must raise; valid segment returns text."""
    assert isinstance(generate_health_advisory(150, "children"), str)
    with pytest.raises(ValueError):
        generate_health_advisory(150, "aliens")


def test_risk_trend_direction():
    """Rising forecast -> worsening; falling -> improving."""
    rising = pd.DataFrame({"forecast": [100, 150, 200]})
    falling = pd.DataFrame({"forecast": [200, 150, 80]})
    flat = pd.DataFrame({"forecast": [100, 102, 101]})
    assert compute_risk_trend(rising) == "worsening"
    assert compute_risk_trend(falling) == "improving"
    assert compute_risk_trend(flat) == "stable"
