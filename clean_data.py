"""
clean_data.py  (Step 1)
-----------------------
Loads the Supabase export and fixes known data problems:
  - keeps automated rows only (manual rows have no weather)
  - drops empty / unhelpful columns
  - turns impossible negative values into NaN
  - fixes wind units (stored as km/h under an "_ms" name)
  - converts pressure to hPa and parses timestamps
"""

import numpy as np
import pandas as pd

import config


def load_and_clean(path=config.DATA_PATH, verbose=True):
    df = pd.read_csv(path, na_values=["null", ""])
    if verbose:
        print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Keep automated rows only
    df = df[df["source"] == "automated"].copy()
    if verbose:
        print(f"Automated rows: {len(df):,}")

    # Drop columns we won't use (ignore any that don't exist)
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    # Negative concentrations / AQIs are invalid -> NaN
    for col in config.NONNEGATIVE_COLS:
        bad = df[col] < 0
        if verbose and bad.sum():
            print(f"  {col}: {bad.sum()} negative values set to NaN")
        df.loc[bad, col] = np.nan

    # Wind: values match METAR knots x 1.852, i.e. km/h. Convert to real m/s.
    df["wind_speed_kmh"] = df["wind_speed_ms"]
    df["wind_speed_ms"] = df["wind_speed_kmh"] / 3.6

    # Pressure in hPa (the unit meteorologists use)
    df["pressure_hpa"] = df["barometric_pressure_pa"] / 100

    # Timestamps (stored in UTC)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)

    dupes = df.duplicated(["city_id", "recorded_at"]).sum()
    if dupes:
        print(f"  WARNING: {dupes} duplicate city-hours; keeping the first of each")
        df = df.drop_duplicates(["city_id", "recorded_at"])

    return df


if __name__ == "__main__":
    # Run this file on its own to check the cleaning step.
    df = load_and_clean()
    cols = config.NONNEGATIVE_COLS + ["temperature_c", "relative_humidity",
                                      "wind_speed_ms", "pressure_hpa"]
    print(df[cols].describe().T.round(2))
