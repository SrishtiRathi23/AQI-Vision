"""Health risk classification using official CPCB AQI bands.

These bands and the advisory logic are domain rules, not learned — they map a
predicted AQI number into an actionable health message for a given population
segment.
"""

from __future__ import annotations

import pandas as pd

import config

# Per-segment advisory text keyed by category label.
_ADVISORIES = {
    "general": {
        "Good": "Air quality is good. Enjoy normal outdoor activities.",
        "Satisfactory": "Air quality is acceptable for most people.",
        "Moderate": "Sensitive individuals should limit prolonged exertion outdoors.",
        "Poor": "Reduce prolonged or heavy outdoor exertion.",
        "Very Poor": "Avoid outdoor exertion; keep windows closed.",
        "Severe": "Stay indoors. Avoid all outdoor activity.",
    },
    "children": {
        "Good": "Safe for outdoor play.",
        "Satisfactory": "Outdoor play is fine.",
        "Moderate": "Watch for coughing/wheezing during play; take breaks.",
        "Poor": "Limit outdoor play; move recess indoors.",
        "Very Poor": "Keep children indoors.",
        "Severe": "Children must remain indoors with air purification if possible.",
    },
    "elderly": {
        "Good": "No precautions needed.",
        "Satisfactory": "No precautions needed.",
        "Moderate": "Limit strenuous outdoor activity.",
        "Poor": "Avoid outdoor exertion; carry prescribed inhalers.",
        "Very Poor": "Remain indoors; monitor cardiac/respiratory symptoms.",
        "Severe": "Stay indoors; seek medical help if symptoms worsen.",
    },
    "respiratory_patients": {
        "Good": "No precautions needed.",
        "Satisfactory": "Keep reliever medication handy.",
        "Moderate": "Reduce outdoor time; pre-medicate before going out.",
        "Poor": "Stay indoors; use prescribed medication proactively.",
        "Very Poor": "Strictly indoors; use air purifier; consult doctor if needed.",
        "Severe": "Medical emergency risk — stay indoors and contact your doctor.",
    },
    "outdoor_workers": {
        "Good": "Normal work conditions.",
        "Satisfactory": "Normal work conditions.",
        "Moderate": "Take regular breaks; stay hydrated.",
        "Poor": "Wear N95 masks; rotate shifts to limit exposure.",
        "Very Poor": "Mandatory N95 masks; shorten outdoor shifts.",
        "Severe": "Suspend outdoor work where possible.",
    },
}

VALID_SEGMENTS = tuple(_ADVISORIES.keys())


def classify_health_risk(aqi_value: float) -> dict:
    """Map an AQI value to its CPCB category, colour and short message.

    Args:
        aqi_value: predicted/observed AQI.

    Returns:
        Dict with category, color (hex) and a general health message.
    """
    aqi = max(0.0, float(aqi_value))
    for low, high, label, color in config.AQI_CATEGORIES:
        if aqi <= high:
            return {
                "aqi": round(aqi, 1),
                "category": label,
                "color": color,
                "message": _ADVISORIES["general"][label],
            }
    # Above 500 -> clamp to the worst band.
    _, _, label, color = config.AQI_CATEGORIES[-1]
    return {"aqi": round(aqi, 1), "category": label, "color": color,
            "message": _ADVISORIES["general"][label]}


def generate_health_advisory(aqi_value: float, population_segment: str = "general") -> str:
    """Return a segment-specific advisory string.

    Args:
        aqi_value: predicted/observed AQI.
        population_segment: one of VALID_SEGMENTS.

    Returns:
        Advisory text tailored to the segment and AQI category.

    Raises:
        ValueError: if the segment is unknown.
    """
    if population_segment not in _ADVISORIES:
        raise ValueError(
            f"Unknown segment '{population_segment}'. Valid: {VALID_SEGMENTS}"
        )
    category = classify_health_risk(aqi_value)["category"]
    return _ADVISORIES[population_segment][category]


def compute_risk_trend(forecast_df: pd.DataFrame, value_col: str = "forecast") -> str:
    """Classify a forecast horizon as improving / worsening / stable.

    Args:
        forecast_df: DataFrame with a forecast column ordered by time.
        value_col: name of the forecast column.

    Returns:
        "improving", "worsening" or "stable" based on first-vs-last comparison
        with a 10% dead-band to avoid flapping on noise.
    """
    values = forecast_df[value_col].to_numpy()
    if len(values) < 2:
        return "stable"
    first, last = values[0], values[-1]
    change = (last - first) / (first + 1e-6)
    if change > 0.10:
        return "worsening"
    if change < -0.10:
        return "improving"
    return "stable"
