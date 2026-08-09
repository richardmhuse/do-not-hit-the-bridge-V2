"""
Train multiple XGBoost models on data/processed/features.csv:

1. Nowcast residual (measured_gauge_height_ft - tide_ft) at time t
   → fills measurement gaps / display lag.

2. Multi-horizon residual regression for lead times
   1 h, 3 h, 6 h, 12 h, 24 h, 36 h.
   Evaluation is TRUE H-hour-ahead: features at t vs residual/level at t+H.

3. Binary classification of threshold crossovers at 1.86 ft:
   - rising  (level crosses upward through 1.86 ft within the next H hours)
   - falling (level crosses downward through 1.86 ft within the next H hours)
   Evaluation uses features at t vs whether a real crossover occurred in (t, t+H].

All models share the same feature pipeline and a time-based hold-out.
A consolidated metrics table is printed at the end for every target.
"""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

from config import DATA_PROCESSED

FEATURES_PATH = DATA_PROCESSED / "features.csv"
MODEL_DIR = DATA_PROCESSED / "model"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HORIZONS_H = [1, 3, 6, 12, 24, 36]
CROSSOVER_THRESH_FT = 1.86
# Hold-out must be longer than the longest horizon so t+36h labels exist
DEFAULT_TEST_DAYS = 5

EXCLUDE = {
    "measured_gauge_height_ft",
    "measured_water_level",
    "measured_Water Level",
    "measured_depth",
    "measured_value",
    "phase_name",
    "short_forecast",
    "wind_direction",
    "station",
    "tide_residual",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_xy(
    df: pd.DataFrame,
    target: str,
    extra_drop: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return X, y and the ordered list of feature names."""
    numeric = df.select_dtypes(include=[np.number]).copy()

    drop = set(EXCLUDE)
    if extra_drop:
        drop.update(extra_drop)
    drop.add(target)

    target_like = [
        c
        for c in numeric.columns
        if c.startswith(("residual_", "cross_rising_", "cross_falling_", "future_level_", "future_tide_"))
        or c == "tide_residual"
    ]
    drop.update(target_like)

    feature_cols = [c for c in numeric.columns if c not in drop]
    X = numeric[feature_cols].copy()
    y = numeric[target].copy()

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    X = X.ffill().bfill()
    for col in X.columns:
        if X[col].isna().any():
            if "rain" in col.lower():
                X[col] = X[col].fillna(0.0)
            else:
                med = X[col].median()
                X[col] = X[col].fillna(med if pd.notna(med) else 0.0)

    still_bad = X.isna().any(axis=1) | y.isna()
    if still_bad.any():
        X = X.loc[~still_bad]
        y = y.loc[~still_bad]

    return X, y, feature_cols


def _infer_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=15)
    diffs = index.to_series().diff().dropna()
    q = diffs.quantile(0.9)
    diffs = diffs[diffs <= q]
    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return pd.Timedelta(minutes=15)
    return med


def _steps_for_horizon(freq: pd.Timedelta, hours: int) -> int:
    return max(1, int(round(pd.Timedelta(hours=hours) / freq)))


def build_crossover_labels(
    level: pd.Series,
    thresh: float,
    steps: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Binary labels: did a rising/falling crossover of `thresh` occur inside
    the *full* next `steps` observations? Incomplete future windows → NaN.
    """
    n = len(level)
    rising = np.full(n, np.nan, dtype=np.float64)
    falling = np.full(n, np.nan, dtype=np.float64)
    vals = level.to_numpy(dtype=float)

    for i in range(n - steps):
        window = vals[i : i + steps + 1]
        if len(window) < steps + 1 or np.any(np.isnan(window)):
            continue

        below = window[:-1] < thresh
        above_or_eq = window[1:] >= thresh
        rising[i] = 1.0 if np.any(below & above_or_eq) else 0.0

        above = window[:-1] > thresh
        below_or_eq = window[1:] <= thresh
        falling[i] = 1.0 if np.any(above & below_or_eq) else 0.0

    idx = level.index
    return (
        pd.Series(rising, index=idx, name="cross_rising"),
        pd.Series(falling, index=idx, name="cross_falling"),
    )


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    cutoff = X.index.max() - pd.Timedelta(days=test_days)
    train_mask = X.index < cutoff
    test_mask = X.index >= cutoff

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    if len(X_test) < 5 or len(X_train) < 20:
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        print("  Short history – using 80/20 sequential split")

    return X_train, y_train, X_test, y_test


def train_regressor(
    X_train, y_train, X_test, y_test, params: dict
) -> Tuple[xgb.XGBRegressor, np.ndarray]:
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    return model, pred


def train_classifier(
    X_train, y_train, X_test, y_test, params: dict
) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return model, pred, proba


def residual_metrics(
    y_true_resid: pd.Series,
    pred_resid: np.ndarray,
    y_true_level: pd.Series | None = None,
    pred_level: np.ndarray | None = None,
) -> dict:
    mae_r = float(mean_absolute_error(y_true_resid, pred_resid))
    rmse_r = float(np.sqrt(mean_squared_error(y_true_resid, pred_resid)))
    out = {
        "residual_mae_ft": mae_r,
        "residual_rmse_ft": rmse_r,
        "n_test": int(len(y_true_resid)),
    }
    if y_true_level is not None and pred_level is not None:
        out["level_mae_ft"] = float(mean_absolute_error(y_true_level, pred_level))
        out["level_rmse_ft"] = float(np.sqrt(mean_squared_error(y_true_level, pred_level)))
    return out


def classifier_metrics(
    y_true: pd.Series, pred: np.ndarray, proba: np.ndarray, scale_pos_weight: float
) -> dict:
    pos = float((y_true == 1).sum())
    neg = float((y_true == 0).sum())
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, proba)) if y_true.nunique() > 1 else 0.0,
        "n_test": int(len(y_true)),
        "pos_rate_test": float(pos / (pos + neg)) if (pos + neg) > 0 else 0.0,
        "scale_pos_weight": scale_pos_weight,
    }
    return metrics


def save_model(
    model,
    name: str,
    feature_cols: List[str],
    metrics: dict,
    task: str,
    extra_meta: dict | None = None,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{name}.json"
    meta_path = MODEL_DIR / f"meta_{name}.json"

    model.save_model(model_path)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "task": task,
        "feature_cols": feature_cols,
        "metrics": metrics,
        **(extra_meta or {}),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  → saved {model_path.name}")


def print_metrics_table(all_metrics: List[dict]):
    """Pretty consolidated report for every trained target."""
    print("\n" + "=" * 78)
    print("HOLD-OUT METRICS SUMMARY  (true lead-time evaluation, not next reading)")
    print("=" * 78)

    # Regression block
    reg = [m for m in all_metrics if m["kind"] == "regression"]
    if reg:
        print("\n  RESIDUAL / LEVEL regression")
        print(
            f"  {'target':<18} {'lead':>6} {'n_test':>7} "
            f"{'resid MAE':>10} {'resid RMSE':>11} "
            f"{'level MAE':>10} {'level RMSE':>11}"
        )
        print("  " + "-" * 74)
        for m in reg:
            lead = f"{m.get('horizon_h', 0)}h" if m.get("horizon_h") else "0h"
            level_mae = m["metrics"].get("level_mae_ft")
            level_rmse = m["metrics"].get("level_rmse_ft")
            print(
                f"  {m['name']:<18} {lead:>6} {m['metrics']['n_test']:>7} "
                f"{m['metrics']['residual_mae_ft']:>10.4f} "
                f"{m['metrics']['residual_rmse_ft']:>11.4f} "
                f"{(f'{level_mae:.4f}' if level_mae is not None else '—'):>10} "
                f"{(f'{level_rmse:.4f}' if level_rmse is not None else '—'):>11}"
            )

    # Classification block
    clf = [m for m in all_metrics if m["kind"] == "classification"]
    if clf:
        print("\n  CROSSOVER classification  (features at t → crossover in (t, t+H])")
        print(
            f"  {'target':<22} {'lead':>6} {'n_test':>7} {'pos%':>6} "
            f"{'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6} {'Brier':>7}"
        )
        print("  " + "-" * 74)
        for m in clf:
            lead = f"{m.get('horizon_h', 0)}h"
            met = m["metrics"]
            print(
                f"  {m['name']:<22} {lead:>6} {met['n_test']:>7} "
                f"{100 * met['pos_rate_test']:>5.1f}% "
                f"{met['accuracy']:>6.3f} {met['precision']:>6.3f} "
                f"{met['recall']:>6.3f} {met['f1']:>6.3f} "
                f"{met['auc']:>6.3f} {met['brier']:>7.4f}"
            )

    print("\n  Notes:")
    print("  • residual_*h  = features at time t, target = residual at t+H  (true H-hour ahead)")
    print("  • level MAE    = |measured(t+H) − (tide(t+H) + predicted_residual)|")
    print("  • cross_*      = features at t, label = did level cross 1.86 ft in (t, t+H]?")
    print("  • Nowcast      = same-time residual (gap-fill only; expect small errors)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    test_days: int = DEFAULT_TEST_DAYS,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
):
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found – run build_features.py first"
        )

    print("Loading features…")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if "tide_ft" not in df.columns:
        raise ValueError("tide_ft missing – cannot train residual model")
    if "measured_gauge_height_ft" not in df.columns:
        raise ValueError("measured_gauge_height_ft missing")

    df["tide_residual"] = df["measured_gauge_height_ft"] - df["tide_ft"]
    df = df.dropna(subset=["tide_residual", "tide_ft", "measured_gauge_height_ft"])

    freq = _infer_freq(df.index)
    print(f"Inferred median sampling interval: {freq}")
    print(f"Hold-out window: last {test_days} days (must exceed longest horizon)")

    all_metrics: List[dict] = []
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }

    # ------------------------------------------------------------------
    # Build targets — residual, future level, future tide, crossovers
    # ------------------------------------------------------------------
    for h in HORIZONS_H:
        steps = _steps_for_horizon(freq, h)
        print(f"  Horizon {h:>2}h → {steps} steps @ {freq}")
        df[f"residual_{h}h"] = df["tide_residual"].shift(-steps)
        df[f"future_level_{h}h"] = df["measured_gauge_height_ft"].shift(-steps)
        df[f"future_tide_{h}h"] = df["tide_ft"].shift(-steps)

    level = df["measured_gauge_height_ft"]
    for h in HORIZONS_H:
        steps = _steps_for_horizon(freq, h)
        rising, falling = build_crossover_labels(level, CROSSOVER_THRESH_FT, steps)
        df[f"cross_rising_{h}h"] = rising
        df[f"cross_falling_{h}h"] = falling

    # ------------------------------------------------------------------
    # 1. Nowcast residual (lead = 0)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("NOWCAST residual  (lead=0 — gap-fill only)")
    print("=" * 60)

    X, y, feature_cols = prepare_xy(df, "tide_residual")
    print(f"Usable rows: {len(X)}  |  features: {len(feature_cols)}")
    if len(X) < 48:
        raise SystemExit(f"Not enough usable rows ({len(X)} < 48)")

    X_train, y_train, X_test, y_test = time_split(X, y, test_days)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    model, pred = train_regressor(X_train, y_train, X_test, y_test, params)

    # Level reconstruction on test: tide_ft already in features/history
    # For nowcast, future_level = measured at same t
    level_true = df.loc[y_test.index, "measured_gauge_height_ft"]
    tide_same = df.loc[y_test.index, "tide_ft"]
    level_pred = tide_same.to_numpy() + pred

    metrics = residual_metrics(y_test, pred, level_true, level_pred)
    metrics["n_train"] = int(len(X_train))
    print(f"  residual MAE={metrics['residual_mae_ft']:.4f}  RMSE={metrics['residual_rmse_ft']:.4f}")
    print(f"  level    MAE={metrics['level_mae_ft']:.4f}  RMSE={metrics['level_rmse_ft']:.4f}")

    importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(8)
    )
    print("  Top features:\n", importance.to_string())

    save_model(
        model,
        "nowcast",
        feature_cols,
        metrics,
        task="nowcast",
        extra_meta={
            "xgb_params": params,
            "test_days": test_days,
            "target": "tide_residual",
            "lead_hours": 0,
            "eval_note": "Same-time residual; NOT a forecast skill score",
        },
    )
    all_metrics.append(
        {"name": "nowcast", "kind": "regression", "horizon_h": 0, "metrics": metrics}
    )

    # Legacy single-model files
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "xgb_model.json")
    legacy_meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": "tide_residual",
        "feature_cols": feature_cols,
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "mae": metrics["residual_mae_ft"],
        "rmse": metrics["residual_rmse_ft"],
        "test_days": test_days,
        "xgb_params": params,
        "note": "Legacy nowcast file; prefer xgb_nowcast.json + multi-horizon models",
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(legacy_meta, f, indent=2)
    print("  → also wrote legacy xgb_model.json + model_meta.json")

    # ------------------------------------------------------------------
    # 2. Multi-horizon residual regression  (TRUE H-hour ahead)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MULTI-HORIZON residual regression  (true H-hour-ahead targets)")
    print("=" * 60)

    for h in HORIZONS_H:
        target = f"residual_{h}h"
        level_col = f"future_level_{h}h"
        tide_col = f"future_tide_{h}h"
        steps = _steps_for_horizon(freq, h)

        print(f"\n--- residual @ t+{h}h  ({steps} steps) ---")
        X, y, feature_cols = prepare_xy(df, target)
        print(f"Usable rows (with full t+{h}h labels): {len(X)}")

        if len(X) < 48:
            print("  Skipping – too few rows")
            continue

        X_train, y_train, X_test, y_test = time_split(X, y, test_days)
        print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

        if len(X_test) < 5:
            print("  Skipping – hold-out too small after shift")
            continue

        model, pred = train_regressor(X_train, y_train, X_test, y_test, params)

        # Absolute level skill at the SAME future timestamps
        level_true = df.loc[y_test.index, level_col]
        tide_future = df.loc[y_test.index, tide_col]
        valid = level_true.notna() & tide_future.notna()
        if valid.any():
            level_pred = tide_future[valid].to_numpy() + pred[valid.to_numpy()]
            metrics = residual_metrics(
                y_test[valid], pred[valid.to_numpy()], level_true[valid], level_pred
            )
        else:
            metrics = residual_metrics(y_test, pred)

        metrics["n_train"] = int(len(X_train))
        metrics["steps"] = steps
        metrics["lead_hours"] = h

        print(
            f"  residual MAE={metrics['residual_mae_ft']:.4f}  "
            f"RMSE={metrics['residual_rmse_ft']:.4f}"
        )
        if "level_mae_ft" in metrics:
            print(
                f"  level    MAE={metrics['level_mae_ft']:.4f}  "
                f"RMSE={metrics['level_rmse_ft']:.4f}   ← operational water-level error at +{h}h"
            )

        save_model(
            model,
            f"residual_{h}h",
            feature_cols,
            metrics,
            task="horizon_residual",
            extra_meta={
                "horizon_hours": h,
                "target": target,
                "xgb_params": params,
                "test_days": test_days,
                "eval_note": (
                    f"Features at t vs residual/level at t+{h}h on hold-out; "
                    "NOT next-reading skill"
                ),
            },
        )
        all_metrics.append(
            {
                "name": f"residual_{h}h",
                "kind": "regression",
                "horizon_h": h,
                "metrics": metrics,
            }
        )

    # ------------------------------------------------------------------
    # 3. Binary crossover classifiers
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CROSSOVER classifiers @ {CROSSOVER_THRESH_FT} ft")
    print("  (features at t → did a real crossover occur in (t, t+H]?)")
    print("=" * 60)

    for h in HORIZONS_H:
        steps = _steps_for_horizon(freq, h)
        for direction, prefix in [
            ("rising", "cross_rising"),
            ("falling", "cross_falling"),
        ]:
            target = f"{prefix}_{h}h"
            print(f"\n--- {direction.upper()} crossover within next {h}h ({steps} steps) ---")

            X, y, feature_cols = prepare_xy(df, target)
            pos_rate = float(y.mean()) if len(y) else 0.0
            print(f"Usable rows: {len(X)}  |  positive rate: {pos_rate:.3f}")

            if len(X) < 48 or y.nunique() < 2:
                print("  Skipping – too few rows or only one class")
                continue

            X_train, y_train, X_test, y_test = time_split(X, y, test_days)
            print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

            if len(X_test) < 5 or y_test.nunique() < 2:
                print("  Skipping – hold-out too small or single-class")
                continue

            model, pred, proba = train_classifier(
                X_train, y_train, X_test, y_test, params
            )
            pos = float((y_train == 1).sum())
            neg = float((y_train == 0).sum())
            scale = neg / pos if pos > 0 else 1.0
            metrics = classifier_metrics(y_test, pred, proba, scale)
            metrics["n_train"] = int(len(X_train))
            metrics["lead_hours"] = h
            metrics["steps"] = steps

            print(
                f"  Acc={metrics['accuracy']:.3f}  "
                f"P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  "
                f"F1={metrics['f1']:.3f}  "
                f"AUC={metrics['auc']:.3f}  "
                f"Brier={metrics['brier']:.4f}"
            )
            print(
                f"  (evaluated on events that actually occurred {h}h after "
                f"each hold-out timestamp — not next reading)"
            )

            save_model(
                model,
                f"{prefix}_{h}h",
                feature_cols,
                metrics,
                task=f"crossover_{direction}",
                extra_meta={
                    "horizon_hours": h,
                    "threshold_ft": CROSSOVER_THRESH_FT,
                    "direction": direction,
                    "target": target,
                    "xgb_params": params,
                    "test_days": test_days,
                    "eval_note": (
                        f"Features at t; label = crossover of {CROSSOVER_THRESH_FT} ft "
                        f"occurred in (t, t+{h}h] on historical hold-out"
                    ),
                },
            )
            all_metrics.append(
                {
                    "name": f"{prefix}_{h}h",
                    "kind": "classification",
                    "horizon_h": h,
                    "metrics": metrics,
                }
            )

    # ------------------------------------------------------------------
    # Consolidated table + index
    # ------------------------------------------------------------------
    print_metrics_table(all_metrics)

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizons_h": HORIZONS_H,
        "crossover_threshold_ft": CROSSOVER_THRESH_FT,
        "test_days": test_days,
        "inferred_freq": str(freq),
        "models": sorted(p.stem for p in MODEL_DIR.glob("xgb_*.json")),
        "metrics": [
            {
                "name": m["name"],
                "kind": m["kind"],
                "horizon_h": m.get("horizon_h"),
                **m["metrics"],
            }
            for m in all_metrics
        ],
    }
    with open(MODEL_DIR / "models_index.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll models written to {MODEL_DIR}")
    print(f"Full metrics also in {MODEL_DIR / 'models_index.json'}")


if __name__ == "__main__":
    main()
