"""
Turn the combined time-series into a model-ready feature matrix.

Datum alignment
---------------
The creek gauge bottoms out near 0 ft (sensor / bed floor). Astronomical
tide (NOAA harmonics) is on a different vertical datum and can go negative
(e.g. below MLLW).  A constant offset is estimated from mid/high water
(where the sensor is not clipped) so that:

    measured_aligned ≈ tide_ft + residual

is well-centered.  Residual models then learn surge/weather error instead of
a fixed vertical mismatch.  The offset is written to
data/processed/datum_offset.json so forecast generation can convert back to
the gauge's native scale for the dashboard.
"""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_PROCESSED

COMBINED_PATH = DATA_PROCESSED / "combined.csv"
FEATURES_PATH = DATA_PROCESSED / "features.csv"
DATUM_OFFSET_PATH = DATA_PROCESSED / "datum_offset.json"

TARGET_CANDIDATES = [
    "measured_gauge_height_ft",
    "measured_water_level",
    "measured_Water Level",
    "measured_depth",
    "measured_value",
]

# Rows at or below this are treated as sensor-floor clipped (unreliable for
# estimating the vertical offset against lunar tide).
SENSOR_FLOOR_FT = 0.30


def find_target(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    measured_cols = [c for c in df.columns if c.startswith("measured_")]
    if not measured_cols:
        raise ValueError("No measured_* column found")
    print(f"Using {measured_cols[0]} as target")
    return measured_cols[0]


def estimate_datum_offset(
    measured: pd.Series,
    tide: pd.Series,
    floor_ft: float = SENSOR_FLOOR_FT,
) -> dict:
    """
    Estimate constant vertical offset so that

        measured ≈ tide + offset + residual

    only using rows where the sensor is clearly above its floor (not clipped).
    Returns offset such that:

        measured_aligned = measured - offset
        residual = measured_aligned - tide   (= measured - tide - offset)

    i.e. offset ≈ median(measured - tide) above the floor.
    """
    both = pd.DataFrame({"m": measured, "t": tide}).dropna()
    if both.empty:
        return {
            "offset_ft": 0.0,
            "method": "none",
            "n_used": 0,
            "floor_ft": floor_ft,
        }

    above = both["m"] > floor_ft
    sample = both.loc[above]
    if len(sample) < 20:
        sample = both  # fall back to everything

    diff = sample["m"] - sample["t"]
    offset_median = float(diff.median())
    offset_mean = float(diff.mean())

    # Robust: also report OLS slope in case datums differ by more than a shift
    A = np.column_stack([sample["t"].to_numpy(), np.ones(len(sample))])
    try:
        coef, *_ = np.linalg.lstsq(A, sample["m"].to_numpy(), rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
    except Exception:
        slope, intercept = 1.0, offset_median

    # Prefer median residual as the operational constant shift (stable, simple)
    offset = offset_median

    return {
        "offset_ft": offset,
        "offset_mean_ft": offset_mean,
        "ols_slope": slope,
        "ols_intercept_ft": intercept,
        "method": "median(measured - tide) where measured > floor",
        "n_used": int(len(sample)),
        "n_total": int(len(both)),
        "floor_ft": floor_ft,
        "measured_min": float(both["m"].min()),
        "measured_max": float(both["m"].max()),
        "tide_min": float(both["t"].min()),
        "tide_max": float(both["t"].max()),
        "pct_floored": float((both["m"] <= floor_ft).mean() * 100.0),
    }


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
    df[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
    df[f"{col}_roll6_mean"] = df[col].rolling(6, min_periods=1).mean()
    df[f"{col}_roll12_mean"] = df[col].rolling(12, min_periods=1).mean()
    df[f"{col}_roll6_std"] = df[col].rolling(6, min_periods=1).std()
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
    if "tide_ft" in df.columns:
        df["tide_ft"] = pd.to_numeric(df["tide_ft"], errors="coerce")

    # ------------------------------------------------------------------
    # Datum alignment: shift measured into the lunar-tide vertical frame
    # ------------------------------------------------------------------
    datum_meta = {
        "offset_ft": 0.0,
        "method": "none",
        "n_used": 0,
        "floor_ft": SENSOR_FLOOR_FT,
    }
    if "tide_ft" in df.columns:
        datum_meta = estimate_datum_offset(df[target], df["tide_ft"], SENSOR_FLOOR_FT)
        offset = datum_meta["offset_ft"]
        print(
            f"Datum offset: {offset:+.4f} ft  "
            f"(median measured−tide above {SENSOR_FLOOR_FT} ft floor, "
            f"n={datum_meta['n_used']})"
        )
        print(
            f"  measured range [{datum_meta['measured_min']:.2f}, {datum_meta['measured_max']:.2f}]  "
            f"tide range [{datum_meta['tide_min']:.2f}, {datum_meta['tide_max']:.2f}]  "
            f"floored {datum_meta['pct_floored']:.1f}%"
        )
        if abs(datum_meta.get("ols_slope", 1.0) - 1.0) > 0.15:
            print(
                f"  ⚠ OLS slope={datum_meta['ols_slope']:.3f} (not ≈1). "
                "A pure vertical shift may be incomplete — check gauge vs NOAA datum."
            )

        # Keep the native gauge reading for the dashboard / thresholds
        df["measured_native_ft"] = df[target]
        # Aligned series lives on the same vertical frame as tide_ft
        df["measured_aligned_ft"] = df[target] - offset
        # Flag sensor-floor clips (residual unreliable here)
        df["is_sensor_floor"] = (df[target] <= SENSOR_FLOOR_FT).astype(int)

        # Residual against lunar tide in the aligned frame
        df["tide_residual"] = df["measured_aligned_ft"] - df["tide_ft"]
        # Equivalent: df[target] - df["tide_ft"] - offset
    else:
        print("No tide_ft column – skipping datum alignment")
        df["measured_native_ft"] = df[target]
        df["measured_aligned_ft"] = df[target]
        df["is_sensor_floor"] = 0

    datum_meta["trained_at"] = datetime.now(timezone.utc).isoformat()
    datum_meta["target_column"] = target
    datum_meta["note"] = (
        "measured_aligned_ft = measured_native_ft - offset_ft. "
        "tide_residual = measured_aligned_ft - tide_ft. "
        "To display a forecast on the gauge scale: "
        "gauge_level = tide_ft + residual + offset_ft."
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(DATUM_OFFSET_PATH, "w") as f:
        json.dump(datum_meta, f, indent=2)
    print(f"Datum meta → {DATUM_OFFSET_PATH}")

    df = add_time_features(df)

    n = len(df)
    possible_lags = [1, 2, 3, 6, 12, 24, 48]
    lags = [lag for lag in possible_lags if lag < n // 3]
    if not lags:
        lags = [1, 2, 3]

    print(f"Using lags: {lags}")
    # Lags / rolls on the *native* measured series (what the sensor reports)
    df = add_lag_features(df, target, lags)
    df = add_rolling_features(df, target)

    if "tide_residual" in df.columns:
        resid_lags = [1, 2, 3, 6, 12, 24]
        resid_lags = [lag for lag in resid_lags if lag < len(df) // 3]
        df = add_lag_features(df, "tide_residual", resid_lags)
        print(
            f"Residual after alignment: mean={df['tide_residual'].mean():.4f}  "
            f"std={df['tide_residual'].std():.4f}  "
            f"median={df['tide_residual'].median():.4f}"
        )

    if "rain_inches" in df.columns:
        df["rain_inches"] = df["rain_inches"].fillna(0.0)
        if "rain_24h" not in df.columns:
            df["rain_24h"] = df["rain_inches"].rolling("24h", min_periods=1).sum()
        if "rain_72h" not in df.columns:
            df["rain_72h"] = df["rain_inches"].rolling("72h", min_periods=1).sum()

    if "wind_speed_mph" in df.columns:
        df["wind_speed_mph"] = df["wind_speed_mph"].ffill()

    if "illumination" in df.columns:
        df["illumination"] = df["illumination"].ffill()

    before = len(df)
    df = df.dropna(subset=[target])
    print(f"Dropped {before - len(df)} rows with missing target")

    short_lags = [c for c in df.columns if c.endswith(("_lag1", "_lag2", "_lag3"))]
    if short_lags:
        df = df.dropna(subset=short_lags)

    df.to_csv(FEATURES_PATH)
    print(f"Features written: {len(df)} rows × {len(df.columns)} columns → {FEATURES_PATH}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
