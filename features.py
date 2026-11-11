"""
features.py  (Step 2)
---------------------
Turns cleaned columns into model inputs:
  2a. cloud cover (oktas) from the cloud_layers JSON
  2b. wind u/v vector components
  2c. local hour / day of week / local date (needs cities.csv timezones)
  2d. lag, rolling-mean, and 24h pressure-change features
  2e. categorical dtypes for LightGBM
"""

import json
import warnings

import numpy as np
import pandas as pd

import config


# ---------- 2a. Cloud cover ----------
def max_cloud_oktas(s):
    """Thickest cloud layer in oktas (0 = clear, 8 = overcast)."""
    if pd.isna(s):
        return np.nan
    try:
        layers = json.loads(s)
    except json.JSONDecodeError:
        return np.nan
    vals = [config.OKTAS[l.get("amount")] for l in layers
            if l.get("amount") in config.OKTAS]
    return max(vals) if vals else np.nan


def add_cloud_cover(df):
    df["cloud_oktas"] = df["cloud_layers"].apply(max_cloud_oktas)
    return df


# ---------- 2b. Wind vectors ----------
def add_wind_vectors(df):
    """Meteorological direction is where wind comes FROM, hence the minus signs."""
    theta = np.deg2rad(df["wind_direction_deg"])
    df["wind_u"] = -df["wind_speed_ms"] * np.sin(theta)   # + = blowing toward east
    df["wind_v"] = -df["wind_speed_ms"] * np.cos(theta)   # + = blowing toward north
    return df


# ---------- 2c. Local time ----------
def add_local_time(df, cities_path=config.CITIES_PATH):
    if cities_path.exists():
        cities = pd.read_csv(cities_path)
        df = df.merge(
            cities[["id", "name", "timezone"]].rename(columns={"id": "city_id"}),
            on="city_id", how="left",
        )
        missing = df["timezone"].isna().sum()
        if missing:
            warnings.warn(f"{missing} rows have no timezone match; using UTC for them")
        df["timezone"] = df["timezone"].fillna("UTC")
    else:
        warnings.warn(
            f"{cities_path} not found. Using UTC for all cities, so local_hour "
            "will be wrong. Export your cities table to fix this."
        )
        df["timezone"] = "UTC"

    df["local_hour"] = np.nan
    df["day_of_week"] = np.nan
    df["local_date"] = None

    # One column can only hold one timezone, so convert one timezone group at a time
    for tz, idx in df.groupby("timezone").groups.items():
        local = df.loc[idx, "recorded_at"].dt.tz_convert(tz)
        df.loc[idx, "local_hour"] = local.dt.hour
        df.loc[idx, "day_of_week"] = local.dt.dayofweek       # Monday = 0
        df.loc[idx, "local_date"] = local.dt.date.astype(str)

    return df


# ---------- 2d. Lags and rolling means ----------
def add_lag_features(df):
    """
    Put each city on a regular hourly grid first, so shift(3) really
    means 3 hours ago even when readings are missing.
    """
    df["is_real"] = True
    lag, roll = config.LAG_HOURS, config.ROLL_HOURS
    pieces = []

    for city, g in df.groupby("city_id"):
        g = g.sort_values("recorded_at").set_index("recorded_at").asfreq("1h")
        g["city_id"] = city                         # filler rows lose their city label

        for col in config.LAG_COLS:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
            g[f"{col}_roll{roll}"] = g[col].rolling(roll, min_periods=config.ROLL_MIN).mean()

        g["pressure_chg24"] = g["pressure_hpa"] - g["pressure_hpa"].shift(24)
        pieces.append(g.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    out = out[out["is_real"].eq(True)].drop(columns="is_real").copy()  # remove filler rows
    return out


# ---------- 2e. Categoricals ----------
def set_categoricals(df):
    for col in config.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def build_features(df, verbose=True):
    n_before = len(df)
    df = add_cloud_cover(df)
    df = add_wind_vectors(df)
    df = add_local_time(df)
    df = add_lag_features(df)
    df = set_categoricals(df)       # must come after the concat in add_lag_features

    if verbose:
        print(f"Features built: {len(df):,} rows (before: {n_before:,})")
    assert len(df) == n_before, "Row count changed during feature engineering"
    return df


if __name__ == "__main__":
    from clean_data import load_and_clean

    df = build_features(load_and_clean(verbose=False))
    new_cols = ["cloud_oktas", "wind_u", "wind_v", "local_hour",
                f"wind_speed_ms_roll{config.ROLL_HOURS}", "pressure_chg24"]
    print(df[new_cols].describe().T.round(2))
