# AQI vs. Weather: LightGBM + SHAP

Measures how much weather explains air quality, and which weather factors
(and combinations) drive each pollutant, using data from the WeatherWebsite archive.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put two files in `data/`:

| File | What it is |
|---|---|
| `data/archive.csv` | The Supabase export of the readings table (rename your download to this) |
| `data/cities.csv`  | Export of the cities table with at least `id`, `name`, `timezone` columns |

Without `cities.csv` the pipeline still runs but uses UTC hours, so `local_hour` is wrong.

## Run

```powershell
python main.py
```

Each stage can also be run alone to check it:

```powershell
python clean_data.py   # Step 1: prints describe() of cleaned columns
python features.py     # Step 2: prints describe() of new features
```

## Files

| File | Step | Purpose |
|---|---|---|
| `config.py` | - | All paths, feature lists, hyperparameters |
| `clean_data.py` | 1 | Load, filter automated rows, fix invalid values and units |
| `features.py` | 2 | Cloud oktas, wind u/v, local time, lags/rolling, categoricals |
| `modeling.py` | 3-5 | Day-grouped CV: baseline vs. weather-only vs. full model |
| `explain.py` | 6 | SHAP importance, dependence plots, interactions |
| `main.py` | all | Runs everything and saves outputs |

## Outputs (`outputs/`)

- `model_comparison.csv`: R² (mean and sd across folds) for baseline, weather-only, and full
  models per target. **`delta_R2` = full − baseline = weather's added explanatory power.**
- `shap_summary.csv`: weather's share of SHAP attribution, top features, top weather pair.
- `<target>/importance_bar.png`, `importance_beeswarm.png`: which features matter, and in which direction.
- `<target>/dependence_<feature>.png`: shape of each top weather effect.
- `<target>/top_interactions_*.csv`: strongest feature pairs.

## Reading delta_R2 (rough guide)

| delta_R2 | Interpretation |
|---|---|
| < 0.05 | Weather adds little beyond city and time |
| 0.05 to 0.15 | Meaningful effect |
| > 0.15 | Major effect |

## Caveats

- Results describe associations learned from the archive's date range (late autumn 2026
  at the start). Summer ozone behavior isn't represented until the archive covers it.
- Hourly readings are strongly autocorrelated; the fold spread (`_sd`) is the honest
  uncertainty, not a p-value.
- `day_of_week` has very few examples per weekday in a short archive; if it ranks high,
  try removing it from `BASELINE` in `config.py`.
- Known data fixes applied in cleaning: wind stored as km/h under `wind_speed_ms`;
  negative sub-AQIs appear where the concentration is missing (likely an ingestion bug).
