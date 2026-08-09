"""
Generate a multi-step forecast past "now" using the trained XGBoost models.

Uses (in priority order):
  1. Direct multi-horizon residual models (residual_1h … residual_36h)
     for accurate level predictions at those exact lead times.
  2. Nowcast / legacy residual model for dense 1-hour recursive fill
     between horizon points and for gap-bridging near "now".
  3. Rising / falling crossover classifiers at 1.86 ft for each horizon.

Writes:
  data/processed/forecast.csv
  data/processed/forecast.json   (compatible with tide_data.py + extra fields)
"""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from config import DATA_PROCESSED, DATA_RAW
except ImportError:
    from config import DATA_PROCESSED
    DATA_RAW = Path("data/raw")

FEATURES_PATH = DATA_PROCESSED / "features.csv"
MODEL_DIR = DATA_PROCESSED / "model"
FORECAST_CSV = DATA_PROCESSED / "forecast.csv"
FORECAST_JSON = DATA_PROCESSED / "forecast.json"
TIDES_PATH = Path(DATA_RAW) / "tides.csv"

HORIZONS_H = [1, 3, 6, 12, 24, 36]
CROSSOVER_THRESH_FT = 1.86
HORIZON_HOURS = 36
STEP_HOURS = 1
BLEND_ALPHA = 0.90  # 1.0 = pure residual, 0.0 = pure tide


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_booster(path: Path, classifier: bool = False):
    if not path.exists():
        return None
    if classifier:
        m = xgb.XGBClassifier()
    else:
        m = xgb.XGBRegressor()
    m.load_model(path)
    return m


def load_models():
    """
    Returns dict:
      nowcast: (model, meta) or None
      residual: {h: (model, meta)}
      cross_rising: {h: (model, meta)}
      cross_falling: {h: (model, meta)}
    """
    models = {
        "nowcast": None,
        "residual": {},
        "cross_rising": {},
        "cross_falling": {},
    }

    # Preferred multi-target nowcast
    nowcast_path = MODEL_DIR / "xgb_nowcast.json"
    nowcast_meta = MODEL_DIR / "meta_nowcast.json"
    if nowcast_path.exists() and nowcast_meta.exists():
        with open(nowcast_meta) as f:
            meta = json.load(f)
        models["nowcast"] = (_load_booster(nowcast_path), meta)
    else:
        # Legacy single model
        legacy_path = MODEL_DIR / "xgb_model.json"
        legacy_meta = MODEL_DIR / "model_meta.json"
        if legacy_path.exists() and legacy_meta.exists():
            with open(legacy_meta) as f:
                meta = json.load(f)
            models["nowcast"] = (_load_booster(legacy_path), meta)

    for h in HORIZONS_H:
        r_path = MODEL_DIR / f"xgb_residual_{h}h.json"
        r_meta = MODEL_DIR / f"meta_residual_{h}h.json"
        if r_path.exists() and r_meta.exists():
            with open(r_meta) as f:
                meta = json.load(f)
            models["residual"][h] = (_load_booster(r_path), meta)

        for direction in ("rising", "falling"):
            c_path = MODEL_DIR / f"xgb_cross_{direction}_{h}h.json"
            c_meta = MODEL_DIR / f"meta_cross_{direction}_{h}h.json"
            if c_path.exists() and c_meta.exists():
                with open(c_meta) as f:
                    meta = json.load(f)
                models[f"cross_{direction}"][h] = (
                    _load_booster(c_path, classifier=True),
                    meta,
                )

    return models


# ---------------------------------------------------------------------------
# Tide helpers
# ---------------------------------------------------------------------------
def load_future_tide(start_time: pd.Timestamp, hours: int = 48) -> pd.Series:
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
    row.loc[row.index[0], col] = value


def build_feature_row(
    hist: pd.DataFrame,
    next_time: pd.Timestamp,
    feature_cols: list[str],
    target: str,
    future_tides: pd.Series,
    last_tide_fallback: float,
) -> pd.DataFrame:
    """Construct a single inference row aligned to feature_cols."""
    row = hist.iloc[[-1]].copy()
    row.index = pd.DatetimeIndex([next_time])

    hour = next_time.hour + next_time.minute / 60.0
    _set(row, "hour_sin", np.sin(2 * np.pi * hour / 24))
    _set(row, "hour_cos", np.cos(2 * np.pi * hour / 24))
    doy = next_time.dayofyear
    _set(row, "doy_sin", np.sin(2 * np.pi * doy / 365.25))
    _set(row, "doy_cos", np.cos(2 * np.pi * doy / 365.25))

    # lags / rolling of residual or measured level
    if "tide_residual" in hist.columns:
        resid_series = hist["tide_residual"].dropna()
    else:
        resid_series = pd.Series(dtype=float)

    for lag in (1, 2, 3, 6, 12, 24, 48):
        col = f"tide_residual_lag{lag}"
        if col in feature_cols and len(resid_series) > 0:
            val = resid_series.iloc[-lag] if len(resid_series) >= lag else resid_series.iloc[-1]
            _set(row, col, val)

    if "measured_gauge_height_ft" in hist.columns:
        level_series = hist["measured_gauge_height_ft"].dropna()
        for lag in (1, 2, 3, 6, 12, 24, 48):
            col = f"measured_gauge_height_ft_lag{lag}"
            if col in feature_cols and len(level_series) > 0:
                val = (
                    level_series.iloc[-lag]
                    if len(level_series) >= lag
                    else level_series.iloc[-1]
                )
                _set(row, col, val)
        for window, suffix in [(3, "roll3_mean"), (6, "roll6_mean"), (12, "roll12_mean")]:
            col = f"measured_gauge_height_ft_{suffix}"
            if col in feature_cols and len(level_series) > 0:
                _set(row, col, float(level_series.iloc[-window:].mean()))
        if "measured_gauge_height_ft_roll6_std" in feature_cols and len(level_series) >= 2:
            _set(row, "measured_gauge_height_ft_roll6_std", float(level_series.iloc[-6:].std()))

    tide_val = tide_at(future_tides, next_time, last_tide_fallback)
    _set(row, "tide_ft", tide_val)

    for col in feature_cols:
        if col not in row.columns or pd.isna(row.iloc[0].get(col, np.nan)):
            if col in hist.columns and hist[col].notna().any():
                _set(row, col, hist[col].dropna().iloc[-1])
            else:
                _set(row, col, 0.0)

    X = pd.DataFrame(
        [[row.iloc[0].get(c, 0.0) for c in feature_cols]],
        columns=feature_cols,
    ).astype(float)
    return X, tide_val, row


# ---------------------------------------------------------------------------
# Dense recursive forecast (nowcast model) for smooth chart line
# ---------------------------------------------------------------------------
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
        X_next, tide_val, row = build_feature_row(
            hist, next_time, feature_cols, target, future_tides, last_tide_fallback
        )
        y_hat = float(model.predict(X_next)[0])

        if is_residual:
            level_hat = tide_val + blend_alpha * y_hat
            residual_hat = y_hat
        else:
            level_hat = y_hat
            residual_hat = level_hat - tide_val

        preds.append({"t": next_time, "predicted": level_hat, "residual": residual_hat})

        new_row = row.copy()
        _set(new_row, target, residual_hat if is_residual else level_hat)
        _set(new_row, "tide_ft", tide_val)
        _set(new_row, "tide_residual", residual_hat)
        if "measured_gauge_height_ft" in hist.columns:
            _set(new_row, "measured_gauge_height_ft", level_hat)
        hist = pd.concat([hist, new_row])

    return pd.DataFrame(preds).set_index("t")


# ---------------------------------------------------------------------------
# Direct multi-horizon residual predictions (preferred when available)
# ---------------------------------------------------------------------------
def direct_horizon_levels(
    residual_models: dict,
    history: pd.DataFrame,
    future_tides: pd.Series,
    last_obs_time: pd.Timestamp,
    blend_alpha: float = BLEND_ALPHA,
) -> dict:
    """
    For each available residual_*h model, predict residual at t+H using
    features from the *current* last observation (no recursive lag update).
    Returns {hours: level_prediction}.
    """
    if "tide_ft" in history.columns and history["tide_ft"].notna().any():
        last_tide_fallback = float(history["tide_ft"].dropna().iloc[-1])
    else:
        last_tide_fallback = 0.0

    out = {}
    for h, (model, meta) in residual_models.items():
        feature_cols = list(meta["feature_cols"])
        next_time = last_obs_time + pd.Timedelta(hours=h)
        X, tide_val, _ = build_feature_row(
            history,
            next_time,
            feature_cols,
            "tide_residual",
            future_tides,
            last_tide_fallback,
        )
        resid_hat = float(model.predict(X)[0])
        level_hat = tide_val + blend_alpha * resid_hat
        out[h] = {
            "t": next_time,
            "predicted": level_hat,
            "residual": resid_hat,
            "tide_ft": tide_val,
        }
    return out


# ---------------------------------------------------------------------------
# Crossover probabilities
# ---------------------------------------------------------------------------
def predict_crossovers(
    models: dict,
    history: pd.DataFrame,
    future_tides: pd.Series,
    last_obs_time: pd.Timestamp,
) -> list[dict]:
    if "tide_ft" in history.columns and history["tide_ft"].notna().any():
        last_tide_fallback = float(history["tide_ft"].dropna().iloc[-1])
    else:
        last_tide_fallback = 0.0

    results = []
    for direction in ("rising", "falling"):
        for h, (model, meta) in models.get(f"cross_{direction}", {}).items():
            feature_cols = list(meta["feature_cols"])
            # Classifiers were trained on features at time t (same row as residual nowcast)
            X, _, _ = build_feature_row(
                history,
                last_obs_time,  # features at "now"
                feature_cols,
                "tide_residual",
                future_tides,
                last_tide_fallback,
            )
            try:
                proba = float(model.predict_proba(X)[0, 1])
            except Exception:
                proba = float(model.predict(X)[0])
            results.append(
                {
                    "direction": direction,
                    "horizon_hours": h,
                    "threshold_ft": meta.get("threshold_ft", CROSSOVER_THRESH_FT),
                    "probability": round(proba, 4),
                    "will_cross": bool(proba >= 0.5),
                }
            )
    # stable order
    results.sort(key=lambda r: (r["direction"], r["horizon_hours"]))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading models…")
    models = load_models()

    if models["nowcast"] is None and not models["residual"]:
        raise FileNotFoundError(
            "No XGBoost models found – run train_xgboost.py first"
        )

    nowcast_model, nowcast_meta = (None, {})
    if models["nowcast"] is not None:
        nowcast_model, nowcast_meta = models["nowcast"]
        print(
            f"Nowcast model loaded  |  target={nowcast_meta.get('target', nowcast_meta.get('task'))}"
        )
    print(f"Direct residual horizons available: {sorted(models['residual'].keys())}")
    print(
        f"Crossover models: rising={sorted(models['cross_rising'].keys())}  "
        f"falling={sorted(models['cross_falling'].keys())}"
    )

    print("Loading latest features…")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()

    # Ensure residual column exists for lag construction
    if "tide_residual" not in df.columns and "measured_gauge_height_ft" in df.columns:
        df["tide_residual"] = df["measured_gauge_height_ft"] - df["tide_ft"]

    history = df.iloc[-48 * 12 :]  # ~48 h at 5-min resolution
    last_obs_time = df.index.max()

    if "measured_gauge_height_ft" in df.columns:
        last_obs_value = float(df["measured_gauge_height_ft"].iloc[-1])
    else:
        last_obs_value = float(df["tide_ft"].iloc[-1] + df.get("tide_residual", pd.Series([0])).iloc[-1])

    now_ts = pd.Timestamp(datetime.now(timezone.utc))
    forecast_end = max(last_obs_time, now_ts) + pd.Timedelta(hours=HORIZON_HOURS)
    total_hours = max(
        float(HORIZON_HOURS),
        (forecast_end - last_obs_time).total_seconds() / 3600.0,
    )
    total_hours = int(np.ceil(total_hours))

    future_tides = load_future_tide(last_obs_time, hours=total_hours + 6)
    print(f"Future tide points available: {len(future_tides)}")

    # --- Dense recursive line (for smooth chart) ---
    if nowcast_model is not None:
        target = nowcast_meta.get("target", "tide_residual")
        feature_cols = list(nowcast_meta["feature_cols"])
        print(
            f"Generating dense recursive forecast from {last_obs_time} "
            f"({total_hours}h, step={STEP_HOURS}h)…"
        )
        forecast = recursive_forecast(
            nowcast_model,
            feature_cols,
            history,
            target,
            future_tides,
            horizon_hours=total_hours,
            step_hours=STEP_HOURS,
            blend_alpha=BLEND_ALPHA,
        )
        forecast = forecast[forecast.index >= last_obs_time]
    else:
        forecast = pd.DataFrame(columns=["predicted", "residual"])

    # --- Direct multi-horizon points (override dense line at exact horizons) ---
    direct = direct_horizon_levels(
        models["residual"], history, future_tides, last_obs_time, BLEND_ALPHA
    )
    if direct:
        print(f"Applying direct residual models at horizons: {sorted(direct.keys())}")
        for h, info in direct.items():
            # Upsert into the dense series
            forecast.loc[info["t"], "predicted"] = info["predicted"]
            forecast.loc[info["t"], "residual"] = info["residual"]
        forecast = forecast.sort_index()

    # --- Crossover predictions ---
    crossovers = predict_crossovers(models, history, future_tides, last_obs_time)
    if crossovers:
        print("Crossover probabilities:")
        for c in crossovers:
            flag = "YES" if c["will_cross"] else "no"
            print(
                f"  {c['direction']:7s}  {c['horizon_hours']:2d}h  "
                f"P={c['probability']:.3f}  → {flag}"
            )

    # Build output frame (stitch last observation)
    out_times = [last_obs_time] + list(forecast.index)
    out_vals = [last_obs_value] + [
        float(v) for v in forecast["predicted"].tolist()
    ]
    out = pd.DataFrame(
        {
            "t": out_times,
            "predicted": out_vals,
            "is_forecast": [False] + [True] * len(forecast),
        }
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(FORECAST_CSV, index=False)

    # Horizon-specific summary for API / UI
    horizon_summary = []
    for h in HORIZONS_H:
        if h in direct:
            horizon_summary.append(
                {
                    "horizon_hours": h,
                    "timestamp": direct[h]["t"].isoformat(),
                    "predicted_level_ft": round(direct[h]["predicted"], 4),
                    "predicted_residual_ft": round(direct[h]["residual"], 4),
                    "source": "direct_residual_model",
                }
            )
        elif not forecast.empty:
            target_t = last_obs_time + pd.Timedelta(hours=h)
            # nearest point in dense forecast
            if target_t in forecast.index:
                val = float(forecast.loc[target_t, "predicted"])
            else:
                # asof
                s = forecast["predicted"].copy()
                s.loc[target_t] = np.nan
                s = s.sort_index().ffill()
                val = float(s.asof(target_t)) if not pd.isna(s.asof(target_t)) else None
            if val is not None:
                horizon_summary.append(
                    {
                        "horizon_hours": h,
                        "timestamp": target_t.isoformat(),
                        "predicted_level_ft": round(val, 4),
                        "source": "recursive_nowcast",
                    }
                )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "blend_alpha": BLEND_ALPHA,
        "model_target": nowcast_meta.get("target", "tide_residual"),
        "predicted_timestamps": [pd.Timestamp(t).isoformat() for t in out["t"]],
        "predicted_values": [float(v) for v in out["predicted"]],
        "model_mae": (nowcast_meta.get("metrics") or {}).get("mae")
        or nowcast_meta.get("mae"),
        "model_rmse": (nowcast_meta.get("metrics") or {}).get("rmse")
        or nowcast_meta.get("rmse"),
        # --- new fields (safe for older tide_data.py; it ignores extras) ---
        "crossover_threshold_ft": CROSSOVER_THRESH_FT,
        "crossovers": crossovers,
        "horizon_summary": horizon_summary,
        "models_used": {
            "nowcast": models["nowcast"] is not None,
            "residual_horizons": sorted(models["residual"].keys()),
            "cross_rising_horizons": sorted(models["cross_rising"].keys()),
            "cross_falling_horizons": sorted(models["cross_falling"].keys()),
        },
    }
    with open(FORECAST_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Forecast written → {FORECAST_CSV}")
    print(f"JSON sidecar   → {FORECAST_JSON}")
    print(f"Points: {len(out)} (1 observed + {len(forecast)} forecast)")


if __name__ == "__main__":
    main()
