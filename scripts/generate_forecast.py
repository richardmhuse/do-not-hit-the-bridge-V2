"""
Generate a multi-step forecast past "now" using trained XGBoost models.

Design (avoids the recursive "strange shape" problem):
  1. Astronomical tide is the base shape (dense, smooth).
  2. Residual corrections come from *direct* horizon models
     (residual_1h … residual_36h) evaluated once on current features —
     never fed back into their own lags.
  3. Residual is interpolated across lead time and blended with a
     lead-dependent alpha (stronger near "now", weaker farther out).
  4. Nowcast model only fills the short gap between last measurement
     and "now" if needed.
  5. Crossover classifiers still run for operational alerts.

Writes:
  data/processed/forecast.csv
  data/processed/forecast.json
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
DATUM_OFFSET_PATH = DATA_PROCESSED / "datum_offset.json"
MODEL_DIR = DATA_PROCESSED / "model"
FORECAST_CSV = DATA_PROCESSED / "forecast.csv"
FORECAST_JSON = DATA_PROCESSED / "forecast.json"
TIDES_PATH = Path(DATA_RAW) / "tides.csv"

HORIZONS_H = [1, 3, 6, 12, 24, 36]
CROSSOVER_THRESH_FT = 1.86
HORIZON_HOURS = 36
# Dense output so the chart doesn't linearly chord hourly samples
OUTPUT_STEP_MINUTES = 15
# Residual trust: full weight near now, decay toward pure tide at long lead
BLEND_ALPHA_NEAR = 0.85   # at lead ≈ 0–1 h
BLEND_ALPHA_FAR = 0.25    # at lead ≥ max horizon
# Astronomical tide at this site runs ~1 h early vs the creek gauge.
# Use tide(t - LAG) as the blend base so highs/lows line up with measurements.
TIDE_LAG_HOURS = 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_booster(path: Path, classifier: bool = False):
    if not path.exists():
        return None
    m = xgb.XGBClassifier() if classifier else xgb.XGBRegressor()
    m.load_model(path)
    return m


def load_models():
    models = {
        "nowcast": None,
        "residual": {},
        "cross_rising": {},
        "cross_falling": {},
    }

    nowcast_path = MODEL_DIR / "xgb_nowcast.json"
    nowcast_meta = MODEL_DIR / "meta_nowcast.json"
    if nowcast_path.exists() and nowcast_meta.exists():
        with open(nowcast_meta) as f:
            meta = json.load(f)
        models["nowcast"] = (_load_booster(nowcast_path), meta)
    else:
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
    start = start_time - pd.Timedelta(hours=6 + TIDE_LAG_HOURS)
    window = tides.loc[(tides.index >= start) & (tides.index <= end), col].astype(float)
    return window


def dense_tide_series(
    future_tides: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_minutes: int = OUTPUT_STEP_MINUTES,
    lag_hours: float = 0.0,
) -> pd.Series:
    """
    Interpolate hourly (or coarser) astronomical tide onto a regular grid.
    Uses time-based linear interpolation so the curve is smooth.

    lag_hours > 0 → at each grid time t, use tide(t - lag), i.e. delay the
    astronomical signal so it lines up with the creek gauge.
    """
    if future_tides is None or future_tides.empty:
        return pd.Series(dtype=float)

    s = future_tides.copy().sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")

    grid = pd.date_range(start=start, end=end, freq=f"{step_minutes}min", tz="UTC")

    if lag_hours and lag_hours != 0:
        # Value shown at t comes from astronomical tide at t - lag
        sample_at = grid - pd.Timedelta(hours=lag_hours)
        combined = s.reindex(s.index.union(sample_at)).sort_index()
        combined = combined.interpolate(method="time").ffill().bfill()
        values = combined.reindex(sample_at).to_numpy()
        return pd.Series(values, index=grid, dtype=float)

    combined = s.reindex(s.index.union(grid)).sort_index()
    combined = combined.interpolate(method="time").ffill().bfill()
    return combined.reindex(grid)


def blend_alpha(lead_hours: float) -> float:
    """Decay residual trust from NEAR → FAR over the forecast horizon."""
    if lead_hours <= 0:
        return BLEND_ALPHA_NEAR
    max_h = float(max(HORIZONS_H))
    frac = min(1.0, lead_hours / max_h)
    return BLEND_ALPHA_NEAR + (BLEND_ALPHA_FAR - BLEND_ALPHA_NEAR) * frac


def load_datum_offset() -> float:
    """
    Vertical offset estimated in build_features.py:

        measured_aligned = measured_native - offset
        residual = measured_aligned - tide_ft

    Forecast on the gauge (native) scale is therefore:

        gauge_level = tide_ft + residual + offset
    """
    if not DATUM_OFFSET_PATH.exists():
        print("  ⚠ datum_offset.json missing – assuming offset=0")
        return 0.0
    try:
        with open(DATUM_OFFSET_PATH) as f:
            meta = json.load(f)
        offset = float(meta.get("offset_ft", 0.0))
        print(f"Datum offset: {offset:+.4f} ft  ({meta.get('method', '?')})")
        return offset
    except Exception as exc:
        print(f"  ⚠ failed to read datum offset ({exc}) – using 0")
        return 0.0


def load_datum_offset() -> float:
    """
    Vertical offset estimated in build_features.py:

        measured_aligned = measured_native - offset
        residual = measured_aligned - tide_ft

    Forecast on the gauge (native) scale is therefore:

        gauge_level = tide_ft + residual + offset
    """
    if not DATUM_OFFSET_PATH.exists():
        print("  ⚠ datum_offset.json missing – assuming offset=0")
        return 0.0
    try:
        with open(DATUM_OFFSET_PATH) as f:
            meta = json.load(f)
        offset = float(meta.get("offset_ft", 0.0))
        print(f"Datum offset: {offset:+.4f} ft  ({meta.get('method', '?')})")
        return offset
    except Exception as exc:
        print(f"  ⚠ failed to read datum offset ({exc}) – using 0")
        return 0.0


# ---------------------------------------------------------------------------
# Feature row for a single inference (no history mutation)
# ---------------------------------------------------------------------------
def _set(row: pd.DataFrame, col: str, value):
    row.loc[row.index[0], col] = value


def build_feature_row(
    hist: pd.DataFrame,
    next_time: pd.Timestamp,
    feature_cols: list[str],
    future_tides: pd.Series,
    last_tide_fallback: float,
) -> tuple[pd.DataFrame, float]:
    """Construct features for `next_time` from *observed* history only."""
    row = hist.iloc[[-1]].copy()
    row.index = pd.DatetimeIndex([next_time])

    hour = next_time.hour + next_time.minute / 60.0
    _set(row, "hour_sin", np.sin(2 * np.pi * hour / 24))
    _set(row, "hour_cos", np.cos(2 * np.pi * hour / 24))
    doy = next_time.dayofyear
    _set(row, "doy_sin", np.sin(2 * np.pi * doy / 365.25))
    _set(row, "doy_cos", np.cos(2 * np.pi * doy / 365.25))

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
            _set(
                row,
                "measured_gauge_height_ft_roll6_std",
                float(level_series.iloc[-6:].std()),
            )

    # Tide at the target time (for residual models that use tide_ft as a feature)
    if future_tides is not None and not future_tides.empty:
        try:
            s = future_tides.copy()
            if next_time not in s.index:
                s.loc[next_time] = np.nan
                s = s.sort_index().interpolate(method="time").ffill().bfill()
            tide_val = float(s.asof(next_time))
            if np.isnan(tide_val):
                tide_val = float(last_tide_fallback)
        except Exception:
            tide_val = float(last_tide_fallback)
    else:
        tide_val = float(last_tide_fallback)
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
    return X, tide_val


# ---------------------------------------------------------------------------
# Direct residual anchors at each trained horizon (no recursion)
# ---------------------------------------------------------------------------
def direct_residual_anchors(
    residual_models: dict,
    nowcast_model_meta: tuple | None,
    history: pd.DataFrame,
    future_tides: pd.Series,
    last_obs_time: pd.Timestamp,
    last_obs_residual: float,
) -> pd.Series:
    """
    Returns a Series indexed by absolute timestamp with residual predictions
    at lead 0 (observed residual) and each available horizon.
    All predictions use the *same* observed feature row — no lag feedback.
    """
    if "tide_ft" in history.columns and history["tide_ft"].notna().any():
        last_tide_fallback = float(history["tide_ft"].dropna().iloc[-1])
    else:
        last_tide_fallback = 0.0

    anchors = {last_obs_time: float(last_obs_residual)}

    # Optional: nowcast residual at "now" (same-time correction)
    if nowcast_model_meta is not None:
        model, meta = nowcast_model_meta
        feature_cols = list(meta["feature_cols"])
        X, _ = build_feature_row(
            history, last_obs_time, feature_cols, future_tides, last_tide_fallback
        )
        try:
            anchors[last_obs_time] = float(model.predict(X)[0])
        except Exception:
            pass

    for h, (model, meta) in residual_models.items():
        feature_cols = list(meta["feature_cols"])
        target_time = last_obs_time + pd.Timedelta(hours=h)
        X, _ = build_feature_row(
            history, target_time, feature_cols, future_tides, last_tide_fallback
        )
        try:
            anchors[target_time] = float(model.predict(X)[0])
        except Exception as exc:
            print(f"  ⚠ residual_{h}h predict failed: {exc}")

    s = pd.Series(anchors, dtype=float).sort_index()
    return s


def residual_on_grid(
    anchors: pd.Series,
    grid: pd.DatetimeIndex,
) -> pd.Series:
    """Time-interpolate residual anchors onto the dense output grid."""
    if anchors.empty:
        return pd.Series(0.0, index=grid)

    s = anchors.copy().sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")

    combined = s.reindex(s.index.union(grid)).sort_index()
    combined = combined.interpolate(method="time").ffill().bfill()
    out = combined.reindex(grid)
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Crossover probabilities (unchanged logic, features at "now")
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
            X, _ = build_feature_row(
                history, last_obs_time, feature_cols, future_tides, last_tide_fallback
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
    results.sort(key=lambda r: (r["direction"], r["horizon_hours"]))
    return results


# ---------------------------------------------------------------------------
# Fallback: pure astronomical tide on dense grid (no ML)
# ---------------------------------------------------------------------------
def pure_tide_forecast(
    future_tides: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bias: float = 0.0,
) -> pd.DataFrame:
    tide = dense_tide_series(future_tides, start, end, lag_hours=TIDE_LAG_HOURS)
    if tide.empty:
        return pd.DataFrame(columns=["predicted", "residual", "tide_ft", "alpha"])
    level = tide + bias
    return pd.DataFrame(
        {
            "predicted": level.values,
            "residual": bias,
            "tide_ft": tide.values,
            "alpha": 0.0,
        },
        index=tide.index,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading models…")
    models = load_models()

    has_any = models["nowcast"] is not None or bool(models["residual"])
    if not has_any:
        print("⚠ No XGBoost models found – will emit pure astronomical tide forecast")

    if models["nowcast"] is not None:
        print(
            f"Nowcast model loaded  |  target="
            f"{models['nowcast'][1].get('target', models['nowcast'][1].get('task'))}"
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

    if "tide_residual" not in df.columns and "measured_gauge_height_ft" in df.columns:
        df["tide_residual"] = df["measured_gauge_height_ft"] - df["tide_ft"]

    history = df.iloc[-48 * 12 :]  # ~48 h at 5-min resolution
    last_obs_time = df.index.max()

    if "measured_gauge_height_ft" in df.columns:
        last_obs_value = float(df["measured_gauge_height_ft"].iloc[-1])
    else:
        last_obs_value = float(
            df["tide_ft"].iloc[-1]
            + df.get("tide_residual", pd.Series([0.0])).iloc[-1]
        )

    if "tide_residual" in df.columns and df["tide_residual"].notna().any():
        last_obs_residual = float(df["tide_residual"].dropna().iloc[-1])
    else:
        last_obs_residual = 0.0

    now_ts = pd.Timestamp(datetime.now(timezone.utc))
    # Forecast from last observation through HORIZON_HOURS past "now"
    forecast_end = max(last_obs_time, now_ts) + pd.Timedelta(hours=HORIZON_HOURS)
    # Start a little before last obs so the stitch is smooth
    forecast_start = last_obs_time

    future_tides = load_future_tide(
        last_obs_time,
        hours=int((forecast_end - last_obs_time).total_seconds() / 3600)
        + 12
        + int(TIDE_LAG_HOURS)
        + 2,
    )
    print(f"Future tide points available: {len(future_tides)}")
    print(
        f"Building dense forecast {forecast_start} → {forecast_end} "
        f"(step={OUTPUT_STEP_MINUTES} min)…"
    )

    # --- Dense astronomical tide (base shape), lagged to match creek timing ---
    print(f"Astronomical tide lag for blend: {TIDE_LAG_HOURS} h")
    tide_dense = dense_tide_series(
        future_tides,
        forecast_start,
        forecast_end,
        lag_hours=TIDE_LAG_HOURS,
    )
    if tide_dense.empty:
        raise RuntimeError(
            "No astronomical tide coverage for the forecast window – "
            "check data/raw/tides.csv"
        )

    grid = tide_dense.index

    # --- Residual anchors from direct horizon models (no recursion) ---
    if models["residual"] or models["nowcast"] is not None:
        anchors = direct_residual_anchors(
            models["residual"],
            models["nowcast"],
            history,
            future_tides,
            last_obs_time,
            last_obs_residual,
        )
        print(f"Residual anchors: {len(anchors)} points at leads "
              f"{sorted(int(round((t - last_obs_time).total_seconds()/3600)) for t in anchors.index)}")
        resid_dense = residual_on_grid(anchors, grid)
    else:
        # Bias = last observed residual held constant
        resid_dense = pd.Series(last_obs_residual, index=grid)
        print("No residual models – holding last observed residual constant")

    # --- Blend: level = tide + alpha(lead) * residual ---
    datum_offset = load_datum_offset()

    leads_h = np.array(
        [(t - last_obs_time).total_seconds() / 3600.0 for t in grid], dtype=float
    )
    alphas = np.array([blend_alpha(h) for h in leads_h], dtype=float)
    residual_vals = resid_dense.to_numpy(dtype=float)
    tide_vals = tide_dense.to_numpy(dtype=float)
    # residual is in the *aligned* frame; add offset back for gauge-native display
    #   gauge_level = tide + alpha * residual + offset
    level_vals = tide_vals + alphas * residual_vals + datum_offset

    # Stitch: force first point to the last measured (native) level for a seamless join
    level_vals = level_vals.copy()
    level_vals[0] = last_obs_value

    forecast = pd.DataFrame(
        {
            "predicted": level_vals,
            "residual": residual_vals,
            "tide_ft": tide_vals,
            "alpha": alphas,
            "lead_hours": leads_h,
        },
        index=grid,
    )

    # --- Crossovers ---
    crossovers = predict_crossovers(models, history, future_tides, last_obs_time)
    if crossovers:
        print("Crossover probabilities:")
        for c in crossovers:
            flag = "YES" if c["will_cross"] else "no"
            print(
                f"  {c['direction']:7s}  {c['horizon_hours']:2d}h  "
                f"P={c['probability']:.3f}  → {flag}"
            )

    # --- Horizon summary (values at exact trained leads) ---
    horizon_summary = []
    for h in HORIZONS_H:
        target_t = last_obs_time + pd.Timedelta(hours=h)
        if target_t in forecast.index:
            row = forecast.loc[target_t]
        else:
            # nearest
            idx = forecast.index.get_indexer([target_t], method="nearest")[0]
            row = forecast.iloc[idx]
            target_t = forecast.index[idx]
        horizon_summary.append(
            {
                "horizon_hours": h,
                "timestamp": target_t.isoformat(),
                "predicted_level_ft": round(float(row["predicted"]), 4),
                "predicted_residual_ft": round(float(row["residual"]), 4),
                "tide_ft": round(float(row["tide_ft"]), 4),
                "alpha": round(float(row["alpha"]), 3),
                "source": (
                    "direct_residual_model"
                    if h in models["residual"]
                    else "interpolated_residual"
                ),
            }
        )

    # Output frame (include last observation as non-forecast stitch point)
    out = pd.DataFrame(
        {
            "t": list(forecast.index),
            "predicted": list(forecast["predicted"]),
            "is_forecast": [False] + [True] * (len(forecast) - 1),
        }
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(FORECAST_CSV, index=False)

    nowcast_meta = models["nowcast"][1] if models["nowcast"] else {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "output_step_minutes": OUTPUT_STEP_MINUTES,
        "blend_alpha_near": BLEND_ALPHA_NEAR,
        "blend_alpha_far": BLEND_ALPHA_FAR,
        "datum_offset_ft": datum_offset,
        "tide_lag_hours": TIDE_LAG_HOURS,
        "model_target": nowcast_meta.get("target", "tide_residual"),
        "predicted_timestamps": [pd.Timestamp(t).isoformat() for t in out["t"]],
        "predicted_values": [float(v) for v in out["predicted"]],
        "model_mae": (nowcast_meta.get("metrics") or {}).get("mae")
        or nowcast_meta.get("mae"),
        "model_rmse": (nowcast_meta.get("metrics") or {}).get("rmse")
        or nowcast_meta.get("rmse"),
        "crossover_threshold_ft": CROSSOVER_THRESH_FT,
        "crossovers": crossovers,
        "horizon_summary": horizon_summary,
        "models_used": {
            "nowcast": models["nowcast"] is not None,
            "residual_horizons": sorted(models["residual"].keys()),
            "cross_rising_horizons": sorted(models["cross_rising"].keys()),
            "cross_falling_horizons": sorted(models["cross_falling"].keys()),
            "method": "dense_tide_plus_interpolated_direct_residual",
        },
    }
    with open(FORECAST_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Forecast written → {FORECAST_CSV}")
    print(f"JSON sidecar   → {FORECAST_JSON}")
    print(
        f"Points: {len(out)} @ {OUTPUT_STEP_MINUTES}-min resolution "
        f"(tide-shaped + residual anchors, no recursive lag feedback)"
    )


if __name__ == "__main__":
    main()
