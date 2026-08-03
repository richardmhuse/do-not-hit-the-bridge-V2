"""
Combine all sources onto the measured gauge timeline (highest frequency).
"""
import pandas as pd
from pathlib import Path

from config import DATA_RAW, DATA_PROCESSED


def load_csv(path: Path, time_col: str = "t") -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ⚠ missing {path} – skipping")
        return None
    df = pd.read_csv(path, parse_dates=[time_col])
    if df[time_col].dt.tz is None:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    else:
        df[time_col] = df[time_col].dt.tz_convert("UTC")
    return df.set_index(time_col).sort_index()


def main():
    print("Loading raw datasets...")

    # --- measured (long → wide) ---
    measured_path = DATA_RAW / "measured.csv"
    if not measured_path.exists():
        raise FileNotFoundError(f"{measured_path} is required")

    raw = pd.read_csv(measured_path, parse_dates=["timestamp"])
    raw = raw.rename(columns={"timestamp": "t"})
    if raw["t"].dt.tz is None:
        # gauge timestamps are Eastern local time
        raw["t"] = (
            pd.to_datetime(raw["t"])
            .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
    else:
        raw["t"] = raw["t"].dt.tz_convert("UTC")

    measured = (
        raw.dropna(subset=["t"])
        .pivot_table(index="t", columns="parameter", values="value", aggfunc="mean")
        .add_prefix("measured_")
        .sort_index()
    )
    print(f"  measured: {len(measured)} rows, columns = {list(measured.columns)}")

    tides   = load_csv(DATA_RAW / "tides.csv")
    weather = load_csv(DATA_RAW / "weather.csv")
    rain    = load_csv(DATA_RAW / "rain.csv")
    lunar   = load_csv(DATA_RAW / "lunar.csv")

    # Start from the dense measured series
    combined = measured.copy()

    def asof_join(left: pd.DataFrame, right: pd.DataFrame | None, name: str, tol="45min"):
        if right is None:
            return left
        out = pd.merge_asof(
            left.sort_index(),
            right.sort_index(),
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta(tol),
        )
        print(f"  joined {name}")
        return out

    combined = asof_join(combined, tides,   "tides")
    combined = asof_join(combined, weather, "weather")
    combined = asof_join(combined, rain,    "rain")
    combined = asof_join(combined, lunar,   "lunar", tol="2h")

    # Clean-ups
    if "rain_inches" in combined.columns:
        combined["rain_inches"] = combined["rain_inches"].fillna(0.0)
        combined["rain_24h"] = combined["rain_inches"].rolling("24h", min_periods=1).sum()
        combined["rain_72h"] = combined["rain_inches"].rolling("72h", min_periods=1).sum()
        combined["rain_7d"]  = combined["rain_inches"].rolling("7d",  min_periods=1).sum()

    if lunar is not None:
        for c in ("illumination", "phase_angle"):
            if c in combined.columns:
                combined[c] = combined[c].ffill()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "combined.csv"
    combined.to_csv(out)
    print(f"\nCombined dataset: {len(combined)} rows → {out}")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
