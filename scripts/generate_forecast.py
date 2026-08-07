"""
Generate a multi-step forecast past "now" using the trained XGBoost model.

Supports two modes (auto-detected from model_meta.json):
  - target == "tide_residual" → residual model + blend with astronomical tide
  - otherwise                 → absolute-level model (legacy behaviour)

Writes:
  data/processed/forecast.csv
  data/processed/forecast.json
"""
from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

# ----- paths (tolerant if config is minimal) -----
try:
    from config import DATA_PROCESSED, DATA_RAW
except ImportError:
    from config import DATA_PROCESSED
    DATA_RAW = Path("data/raw")

FEATURES_PATH = DATA_PROCESSED / "features.csv"
MODEL_PATH = DATA_PROCESSED / "model" / "xgb_model.json"
META_PATH = DATA_PROCESSED / "model" / "model_meta.json"
FORECAST_CSV = DATA_PROCESSED / "forecast.csv"
FORECAST_JSON = DATA_PROCESSED / "forecast.json"
TIDES_PATH = Path(DATA_RAW) / "tides.csv"

HORIZON_HOURS = 36
STEP_HOURS = 1
BLEND_ALPHA = 0.8  # 1.0 = pure residual, 0.0 = pure tide


def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model not found – run train_xgboost.py first")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


def load_future_tide(start_time: pd.Timestamp, hours: int = 24) -> pd.Series:
    """Return tide_ft series covering roughly [start_time, start_time + hours]."""
    if not TIDES_PATH.exists():
        print(f"  ⚠ tides file missing: {TIDES_PATH}")
        return pd.Series(dtype=float)

    tides = pd.read_csv(TIDES_PATH, parse_dates=["t"])
    if "t" not in tides.columns:
        print("  ⚠ tides.csv has no 't' column")
        return pd.Series(dtype=float)

    t = pd.to_datetime(tides["t"], utc=True, errors="coerce")
    tides = tides.assign(t=t).dropna(subset=["t"]).set_index("t").sort_index()

    col = "tide_ft" if "tide_ft" in tides.columns else ("v" if "v" in tides.columns else None)
    if col is None:
        print("  ⚠ tides.csv has no tide_ft / v column")
        return pd.Series(dtype=float)

    end = start_time + pd.Timedelta(hours=hours)
    start = start_time - pd.Timedelta(hours=2)
    window = tides.loc[(tides.index >= start) & (tides.index <= end), col].astype(float)
    return window


def tide_at(series: pd.Series, ts: pd.Timestamp, fallback: float) -> float:
    if series is None or series.empty:
        return float(fallback)
    try:
        # prefer exact / nearest via reindex + ffill/bfill
        s = series.copy()
        if ts not in s.index:
            s.loc[ts] = np.nan
            s = s.sort_index().ffill().bfill()
        val = s.asof(ts)
        if pd.isna(val):
            return float(fallback)
        return float(val)
    except Exception:
        return float(fallback)


def _set(row: pd.DataFrame, col: str, value):
    """Safe scalar assign into a one-row DataFrame."""
    row.loc[row.index[0], col] = value


def recursive_forecast(
    model,
    feature_cols: list[str],
    history: pd.DataFrame,
    target: str,
    future_tides: pd.Series,
    horizon_hours: int = HORIZON_HOURS,
    step_hours: int = STEP_HOURS,
    blend_alpha: float = BLEND_ALPHA,
) -> pd.DataFrame:
    hist = history.copy().sort_index()
    last_time = hist.index.max()

    if "tide_ft" in hist.columns and hist["tide_ft"].notna().any():
        last_tide_fallback = float(hist["tide_ft"].dropna().iloc[-1])
    else:
        last_tide_fallback = 0.0

    is_residual = target == "tide_residual"
    preds = []

    for step in range(1, int(horizon_hours / step_hours) + 1):
        next_time = last_time + pd.Timedelta(hours=step_hours * step)

        row = hist.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex([next_time])

        # time features
        hour = next_time.hour + next_time.minute / 60.0
        _set(row, "hour_sin", np.sin(2 * np.pi * hour / 24))
        _set(row, "hour_cos", np.cos(2 * np.pi * hour / 24))
        doy = next_time.dayofyear
        _set(row, "doy_sin", np.sin(2 * np.pi * doy / 365.25))
        _set(row, "doy_cos", np.cos(2 * np.pi * doy / 365.25))

        # lags / rolling of the training target
        if target in hist.columns:
            target_series = hist[target].dropna()
        else:
            target_series = pd.Series(dtype=float)

        for lag in (1, 2, 3, 6, 12, 24, 48):
            col = f"{target}_lag{lag}"
            if col in feature_cols and len(target_series) > 0:
                val = (
                    target_series.iloc[-lag]
                    if len(target_series) >= lag
                    else target_series.iloc[-1]
                )
                _set(row, col, val)

        for window, suffix in [(3, "roll3_mean"), (6, "roll6_mean"), (12, "roll12_mean")]:
            col = f"{target}_{suffix}"
            if col in feature_cols and len(target_series) > 0:
                _set(row, col, float(target_series.iloc[-window:].mean()))

        if f"{target}_roll6_std" in feature_cols and len(target_series) >= 2:
            _set(row, f"{target}_roll6_std", float(target_series.iloc[-6:].std()))

        # residual lags
        if "tide_residual" in hist.columns:
            resid_series = hist["tide_residual"].dropna()
            for lag in (1, 2, 3, 6, 12, 24):
                col = f"tide_residual_lag{lag}"
                if col in feature_cols and len(resid_series) > 0:
                    val = (
                        resid_series.iloc[-lag]
                        if len(resid_series) >= lag
                        else resid_series.iloc[-1]
                    )
                    _set(row, col, val)

        # future tide
        tide_val = tide_at(future_tides, next_time, last_tide_fallback)
        _set(row, "tide_ft", tide_val)

        # fill any remaining feature_cols from last history value
        for col in feature_cols:
            if col not in row.columns or pd.isna(row.iloc[0].get(col, np.nan)):
                if col in hist.columns and hist[col].notna().any():
                    _set(row, col, hist[col].dropna().iloc[-1])
                else:
                    _set(row, col, 0.0)

        # build X in exact training column order
        X_next = pd.DataFrame(
            [[row.iloc[0].get(c, 0.0) for c in feature_cols]],
            columns=feature_cols,
        ).astype(float)

        y_hat = float(model.predict(X_next)[0])

        if is_residual:
            level_hat = tide_val + blend_alpha * y_hat
            residual_hat = y_hat
        else:
            level_hat = y_hat
            residual_hat = level_hat - tide_val

        preds.append({"t": next_time, "predicted": level_hat})

        # append to history for next lags
        new_row = row.copy()
        _set(new_row, target, residual_hat if is_residual else level_hat)
        _set(new_row, "tide_ft", tide_val)
        _set(new_row, "tide_residual", residual_hat)
        if "measured_gauge_height_ft" in hist.columns:
            _set(new_row, "measured_gauge_height_ft", level_hat)

        hist = pd.concat([hist, new_row])

    return pd.DataFrame(preds).set_index("t")


def main():
    print("Loading model…")
    model, meta = load_model_and_meta()
    target = meta["target"]
    feature_cols = list(meta["feature_cols"])
    is_residual = target == "tide_residual"
    print(
        f"Model target: {target}  |  residual mode: {is_residual}  |  blend α={BLEND_ALPHA}"
    )

    print("Loading latest features…")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    history = df.iloc[-48 * 12 :]  # ~48 h at 5‑min resolution
    last_obs_time = df.index.max()

    # stitch point must be absolute water level
    if "measured_gauge_height_ft" in df.columns:
        last_obs_value = float(df["measured_gauge_height_ft"].iloc[-1])
    elif not is_residual:
        last_obs_value = float(df[target].iloc[-1])
    else:
        last_obs_value = float(df["tide_ft"].iloc[-1] + df[target].iloc[-1])

    now_ts = pd.Timestamp(datetime.now(timezone.utc))
    # Always project far enough that the line ends at least HORIZON_HOURS past "now"
    forecast_end = max(last_obs_time, now_ts) + pd.Timedelta(hours=HORIZON_HOURS)
    total_hours = max(
        float(HORIZON_HOURS),
        (forecast_end - last_obs_time).total_seconds() / 3600.0,
    )
    total_hours = int(np.ceil(total_hours))

    future_tides = load_future_tide(last_obs_time, hours=total_hours + 6)
    print(f"Future tide points available: {len(future_tides)}")
    print(
        f"Generating forecast from {last_obs_time} → {forecast_end} "
        f"({total_hours}h span, step={STEP_HOURS}h)…"
    )

    forecast = recursive_forecast(
        model,
        feature_cols,
        history,
        target,
        future_tides,
        horizon_hours=total_hours,
        step_hours=STEP_HOURS,
        blend_alpha=BLEND_ALPHA,
    )

    # Keep points from last observation onward (smooth stitch on the chart)
    forecast = forecast[forecast.index >= last_obs_time]

    out = pd.DataFrame(
        {
            "t": [last_obs_time] + list(forecast.index),
            "predicted": [last_obs_value] + list(forecast["predicted"]),
            "is_forecast": [False] + [True] * len(forecast),
        }
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(FORECAST_CSV, index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "blend_alpha": BLEND_ALPHA,
        "model_target": target,
        "predicted_timestamps": [pd.Timestamp(t).isoformat() for t in out["t"]],
        "predicted_values": [float(v) for v in out["predicted"]],
        "model_mae": meta.get("mae"),
        "model_rmse": meta.get("rmse"),
    }
    with open(FORECAST_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Forecast written → {FORECAST_CSV}")
    print(f"JSON sidecar   → {FORECAST_JSON}")
    print(f"Points: {len(out)} (1 observed + {len(forecast)} forecast)")


if __name__ == "__main__":
    main()
