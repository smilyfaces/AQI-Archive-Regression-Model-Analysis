"""
config.py
---------
Every setting for the project lives here: file paths, feature lists,
model hyperparameters, and SHAP sample sizes. Change things here
rather than editing the other files.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths (relative to this file, so the project works from any folder)
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH   = PROJECT_DIR / "data" / "archive.csv"   # your Supabase export
CITIES_PATH = PROJECT_DIR / "data" / "cities.csv"    # needs columns: id, name, timezone
OUTPUT_DIR  = PROJECT_DIR / "outputs"

# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------
# Columns that are empty, almost empty, raw text, or derived from others.
DROP_COLS = [
    "id", "source", "raw", "raw_metar", "created_at",
    "aqi_co",                                            # every value is -999
    "cloud_coverage_pct", "cloud_coverage_label",        # 100% empty
    "precipitation_last_hour_mm", "precipitation_3h_mm",
    "precipitation_6h_mm",                               # >98% empty
    "wind_gust_ms", "sea_level_pressure_pa",             # >90% empty
    "heat_index_c", "wind_chill_c",                      # derived from temp/RH/wind
    "present_weather", "weather_observed_at",
]

# Concentrations and AQI values can never be negative; negatives become NaN.
NONNEGATIVE_COLS = [
    "pm25", "pm10", "ozone", "no2", "so2", "co",
    "aqi_pm25", "aqi_pm10", "aqi_ozone", "aqi_no2", "aqi_so2", "aqi",
]

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
# Sky-cover codes -> oktas (eighths of the sky covered; midpoints for ranges)
OKTAS = {"SKC": 0, "CLR": 0, "FEW": 1.5, "SCT": 3.5, "BKN": 6, "OVC": 8, "VV": 8}

# Columns that get 3-hour lag and 6-hour rolling-mean versions
LAG_COLS    = ["wind_speed_ms", "relative_humidity", "temperature_c"]
LAG_HOURS   = 3
ROLL_HOURS  = 6
ROLL_MIN    = 4      # rolling mean needs at least this many non-missing hours

CATEGORICAL_COLS = ["city_id", "precipitation_type"]

# ---------------------------------------------------------------------
# Feature groups and targets
# ---------------------------------------------------------------------
BASELINE = ["city_id", "local_hour", "day_of_week"]

WEATHER = [
    "temperature_c", "dewpoint_c", "relative_humidity", "wind_speed_ms",
    "wind_u", "wind_v", "pressure_hpa", "cloud_oktas", "precipitation_type",
    *[f"{c}_lag{LAG_HOURS}" for c in LAG_COLS],
    *[f"{c}_roll{ROLL_HOURS}" for c in LAG_COLS],
    "pressure_chg24",
]

TARGETS = ["pm25", "pm10", "ozone", "no2", "aqi"]

# ---------------------------------------------------------------------
# Model and validation
# ---------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,      # small steps: each tree only nudges predictions
    num_leaves=31,           # max leaves per tree (complexity)
    min_child_samples=50,    # each leaf needs >= 50 rows (prevents memorizing)
    subsample=0.8,           # each tree sees 80% of rows...
    subsample_freq=1,
    colsample_bytree=0.8,    # ...and 80% of features
    random_state=42,
    verbose=-1,
)

N_SPLITS     = 5          # cross-validation folds, grouped by local calendar day
GROUP_COL    = "local_date"

# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
SHAP_SAMPLE        = 3000   # rows used for SHAP values (speed)
INTERACTION_SAMPLE = 1000   # rows used for interaction values (much slower)
TOP_PAIRS          = 10     # interaction pairs to report per target
RANDOM_STATE       = 0
