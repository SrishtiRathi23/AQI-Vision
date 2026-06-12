"""Central configuration for the AQI-Vision project.

Every tunable constant lives here so that no module contains "magic numbers".
Keeping them in one place is also an interview talking point: it makes the
pipeline reproducible and the experiment easy to audit.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"

CITY_DAY_FILE = RAW_DIR / "city_day.csv"
CITY_HOUR_FILE = RAW_DIR / "city_hour.csv"
PROCESSED_FILE = PROCESSED_DIR / "features.parquet"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURE_LIST_FILE = MODELS_DIR / "feature_columns.json"
SHAP_VALUES_FILE = MODELS_DIR / "shap_values.pkl"

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
# The 12 raw pollutant columns present in the CPCB "Air Quality in India" data.
POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
TARGET_COL = "AQI"
DATE_COL = "Datetime"        # hourly file uses 'Datetime', daily uses 'Date'
CITY_COL = "City"

# The 26 cities present in the real dataset. Used by the synthetic generator
# and to validate the real file.
CITIES = [
    "Ahmedabad", "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal",
    "Brajrajnagar", "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Ernakulam",
    "Gurugram", "Guwahati", "Hyderabad", "Jaipur", "Jorapokhar", "Kochi",
    "Kolkata", "Lucknow", "Mumbai", "Patna", "Shillong", "Talcher",
    "Thiruvananthapuram", "Visakhapatnam",
]

# Tier-1 metros tend to have systematically higher baseline pollution; we encode
# this as a feature (domain knowledge the model cannot infer from city names).
TIER_1_CITIES = {"Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad"}

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
SHORT_GAP_HOURS = 3          # <= 3 missing hours -> forward fill
MEDIUM_GAP_HOURS = 7         # 4-7 missing hours  -> linear interpolation
IQR_MULTIPLIER = 1.5         # Tukey fence for outlier capping
TEST_SIZE = 0.2              # fraction of the (time-ordered) tail held out

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
LAG_HOURS = [1, 3, 6, 12, 24]
ROLLING_MEAN_WINDOWS = [3, 6, 12, 24]
ROLLING_STD_WINDOWS = [6, 24]
EWMA_SPANS = [6, 24]

# India-specific seasons (month -> season). Standard IMD-style buckets.
SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
    12: "Winter",
}
# Diwali falls in Oct/Nov and is associated with firecracker-driven PM spikes.
FESTIVAL_MONTHS = {10, 11}

# --------------------------------------------------------------------------- #
# Models / training
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_CV_SPLITS = 5              # folds for TimeSeriesSplit
N_RANDOM_SEARCH_ITER = 20   # RandomizedSearchCV sampling budget

RIDGE_PARAMS = {"alpha": 1.0, "random_state": RANDOM_STATE}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 18,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# Search spaces for RandomizedSearchCV (XGBoost / LightGBM).
XGB_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
LGBM_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

SARIMA_ORDER = (1, 1, 1)            # (p, d, q)
SARIMA_SEASONAL_ORDER = (1, 0, 1, 24)  # daily seasonality on hourly data

# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
FORECAST_HORIZONS = [24, 48, 72]
BOOTSTRAP_SAMPLES = 200            # for forecast uncertainty bands

# --------------------------------------------------------------------------- #
# Health risk (official CPCB AQI bands)
# --------------------------------------------------------------------------- #
# (lower_bound_inclusive, upper_bound_inclusive, label, hex_color)
AQI_CATEGORIES = [
    (0, 50, "Good", "#2ECC71"),
    (51, 100, "Satisfactory", "#A3D977"),
    (101, 200, "Moderate", "#F1C40F"),
    (201, 300, "Poor", "#E67E22"),
    (301, 400, "Very Poor", "#E74C3C"),
    (401, 500, "Severe", "#7B241C"),
]

MLFLOW_EXPERIMENT = "aqi-vision-forecasting"
