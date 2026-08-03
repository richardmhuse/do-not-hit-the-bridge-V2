"""
Turn the combined time-series into a model-ready feature matrix.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA_PROCESSED

COMBINED_PATH = DATA_PROCESSED / "combined.csv"
FEATURES_PATH = DATA_PROCESSED / "features.csv"

TARGET_CANDIDATES = [
    "measured_gauge_height_ft",
    "measured_water_level",
    "measured_Water Level",
    "measured_depth",
    "measured_value",
]


def find_target(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    measured_cols = [c for c in df.columns if c.startswith("measured_")]
    if not measured_cols:
        raise ValueError("No measured_* column found")
    print(f"Using {measured_cols[0]} as target")
    return measured_cols[0]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"{col}_roll3_mean"]  = df[col].rolling(3,  min_periods=1).mean()
    df[f"{col}_roll6_mean"]  = df[col].rolling(6,  min_periods=1).mean()
    df[f"{col}_roll12_mean"] = df[col].rolling(12, min_periods=1).mean()
    df[f"{col}_roll6_std"]   = df[col].rolling(6,  min_periods=1).std()
    return df


def main():
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(f"{COMBINED_PATH} not found")

    df = pd.read_csv(COMBINED_PATH, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    target = find_target(df)
    print(f"Target column: {target}")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = add_time_features(df)

    # Adaptive lags: only request lags that the series can actually support
    n = len(df)
    possible_lags = [1, 2, 3, 6, 12, 24, 48]
    lags = [lag for lag in possible_lags if lag < n // 3]   # need some headroom
    if not lags:
        lags = [1, 2, 3]

    print(f"Using lags: {lags}")
    df = add_lag_features(df, target, lags)
    df = add_rolling_features(df, target)

    if "tide_ft" in df.columns:
        df["tide_residual"] = df[target] - df["tide_ft"]
        # lags of the residual (these become the main autoregressive signal)
        resid_lags = [1, 2, 3, 6, 12, 24]
        # only keep lags the series can support
        resid_lags = [lag for lag in resid_lags if lag < len(df) // 3]
        df = add_lag_features(df, "tide_residual", resid_lags)

    if "rain_inches" in df.columns:
        df["rain_inches"] = df["rain_inches"].fillna(0.0)
        if "rain_24h" not in df.columns:
            df["rain_24h"] = df["rain_inches"].rolling("24h", min_periods=1).sum()
        if "rain_72h" not in df.columns:
            df["rain_72h"] = df["rain_inches"].rolling("72h", min_periods=1).sum()

    if "wind_speed_mph" in df.columns:
        df["wind_speed_mph"] = df["wind_speed_mph"].ffill()   # fixed deprecation

    if "illumination" in df.columns:
        df["illumination"] = df["illumination"].ffill()

    before = len(df)
    df = df.dropna(subset=[target])
    print(f"Dropped {before - len(df)} rows with missing target")

    # Only drop rows that are missing the *shortest* lags (keep as much data as possible)
    short_lags = [c for c in df.columns if c.endswith(("_lag1", "_lag2", "_lag3"))]
    if short_lags:
        df = df.dropna(subset=short_lags)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH)
    print(f"Features written: {len(df)} rows × {len(df.columns)} columns → {FEATURES_PATH}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
